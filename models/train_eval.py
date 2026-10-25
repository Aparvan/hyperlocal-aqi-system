"""
Training, Validation, and Evaluation Benchmark Pipeline.
Trains:
1. Baselines (Nearest Station, IDW, Tabular MLP)
2. Multimodal Fusion Deep Learning Network
Computes evaluation metrics: RMSE, MAE, R², and Pearson r across all models.
Saves model weights and benchmark comparison report.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import numpy as np

from config import MODELS_DIR, DATA_DIR, MODEL_CONFIG, DEFAULT_REGION, GROUND_STATIONS
from data.generator import AtmosphericSimulator
from data.preprocessor import HyperlocalPreprocessor, MultimodalDataset
from models.baselines import IDWInterpolator, NearestStationPredictor, TabularMLPBaseline, compute_metrics
from models.multimodal_fusion import MultimodalFusionNetwork
from models.pinn_loss import PhysicsInformedLoss


def run_training_pipeline(epochs: int = 15, batch_size: int = 32, seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 1. Ensure Training Dataset Exists
    dataset_file = DATA_DIR / "training_dataset.json"
    sim = AtmosphericSimulator(DEFAULT_REGION, seed=seed)
    if not dataset_file.exists():
        print("Dataset not found. Generating 4-day atmospheric sequence...")
        sim.generate_training_dataset(num_days=4, samples_per_hour=15)
        
    with open(dataset_file, "r", encoding="utf-8") as f:
        records = json.load(f)
        
    print(f"Loaded {len(records)} records for training and validation.")
    
    # 2. Build Preprocessor and True Spatial Hold-Out Split (Leave-Stations-Out)
    # Holding out 3 distinct geographic station zones (Dwarka, Noida Sec 62, Bawana)
    # ensures zero spatial leakage: model is evaluated strictly on unseen micro-climates.
    prep = HyperlocalPreprocessor(DEFAULT_REGION)
    holdout_station_ids = {"DL_005", "UP_001", "DL_008"}
    
    stations_snapshot = prep.station_fetcher.fetch_all_stations()
    train_stations = [s for s in stations_snapshot if s["station_id"] not in holdout_station_ids]
    
    train_records = []
    val_records = []
    for rec in records:
        nearest_st, _ = prep.station_fetcher.find_nearest_station(rec["lat"], rec["lon"])
        st_id = nearest_st["station_id"] if nearest_st else ""
        if st_id in holdout_station_ids:
            val_records.append(rec)
        else:
            train_records.append(rec)
            
    print(f"Spatial Hold-Out Split: {len(train_records)} train records | {len(val_records)} validation records (Held out stations: {holdout_station_ids})")
    
    # Crucial: pass allowed_stations=train_stations to BOTH train and validation sets!
    # This guarantees that validation records around Dwarka/Bawana/Noida NEVER search
    # or access the held-out station's own readings; nearest station is picked strictly from train_stations.
    train_set = MultimodalDataset(train_records, prep, allowed_stations=train_stations)
    val_set = MultimodalDataset(val_records, prep, allowed_stations=train_stations)
    
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # 3. Train Tabular Baseline MLP
    print("\n--- Training Tabular MLP Baseline ---")
    tabular_model = TabularMLPBaseline(in_features=12, out_features=4).to(device)
    tab_optimizer = torch.optim.AdamW(tabular_model.parameters(), lr=0.003, weight_decay=1e-4)
    tab_criterion = nn.SmoothL1Loss()
    
    for epoch in range(epochs):
        tabular_model.train()
        for batch in train_loader:
            tab_in = batch["tabular"].to(device)
            targets = batch["target"].to(device)
            tab_optimizer.zero_grad()
            preds = tabular_model(tab_in)
            loss = tab_criterion(preds, targets)
            loss.backward()
            tab_optimizer.step()
            
    # 4. Train Multimodal Deep Learning Fusion Network with Physics-Informed (PINN) Loss
    print("\n--- Training Multimodal Deep Learning Fusion Network (with PINN Loss) ---")
    model = MultimodalFusionNetwork(
        spatial_channels=MODEL_CONFIG["spatial_channels"],
        temporal_features=MODEL_CONFIG["temporal_features"],
        tabular_features=MODEL_CONFIG["tabular_features"],
        embedding_dim=32,
        num_heads=4,
        dropout=MODEL_CONFIG["dropout"],
    ).to(device)
    
    pinn_loss_fn = PhysicsInformedLoss(dx_meters=2000.0, dy_meters=2000.0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=MODEL_CONFIG["learning_rate"], weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    criterion = nn.SmoothL1Loss()
    
    best_val_loss = float("inf")
    model_save_path = MODELS_DIR / "multimodal_model.pt"
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            spat = batch["spatial"].to(device)
            temp = batch["temporal"].to(device)
            tab = batch["tabular"].to(device)
            targets = batch["target"].to(device)
            
            optimizer.zero_grad()
            out = model(spat, temp, tab)
            pred_loss = criterion(out["preds"], targets)
            
            # Physics-Informed (PINN) Advection-Diffusion Regularizer
            # Evaluates spatial continuity and wind-transport residual over the model's predicted 2D spatial surface
            # Unnormalize wind vectors to real physical meters/sec
            raw_u = float(tab[0, 7].item()) * 2.0 + 0.0
            raw_v = float(tab[0, 8].item()) * 2.5 - 1.5
            pinn_res = pinn_loss_fn(out["spatial_patch_pred"], wind_u=raw_u, wind_v=raw_v)
            pinn_loss = pinn_res["pinn_loss"]
            
            # Consistency between center of predicted spatial patch and point PM2.5 prediction
            patch_center_loss = criterion(out["spatial_patch_pred"][:, 0, 2, 2], targets[:, 0])
            total_loss = pred_loss + 0.02 * pinn_loss + 0.05 * patch_center_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            train_loss += total_loss.item() * spat.size(0)
            
        train_loss /= len(train_set)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                spat = batch["spatial"].to(device)
                temp = batch["temporal"].to(device)
                tab = batch["tabular"].to(device)
                targets = batch["target"].to(device)
                out = model(spat, temp, tab)
                loss = criterion(out["preds"], targets)
                val_loss += loss.item() * spat.size(0)
                
        val_loss /= len(val_set)
        scheduler.step(val_loss)
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_save_path)
            
        if epoch % 3 == 0 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            
    print(f"\nBest model saved to {model_save_path}")
    
    # 5. Comprehensive Benchmark Evaluation on the EXACT SAME Spatial Hold-Out Set
    print("\n--- Running Comparative Benchmark Evaluation on Spatial Hold-Out Set ---")
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    model.eval()
    tabular_model.eval()
    
    stations_snapshot = prep.station_fetcher.fetch_all_stations()
    # Fit spatial baseline interpolators on training stations only
    train_stations = [s for s in stations_snapshot if s["station_id"] not in holdout_station_ids]
    idw = IDWInterpolator(power=2.0)
    idw.fit(train_stations)
    nearest = NearestStationPredictor()
    nearest.fit(train_stations)
    
    all_y_true = []
    all_pred_multimodal = []
    all_pred_tabular = []
    all_pred_idw = []
    all_pred_nearest = []
    
    with torch.no_grad():
        for batch in val_loader:
            spat = batch["spatial"].to(device)
            temp = batch["temporal"].to(device)
            tab = batch["tabular"].to(device)
            targets = batch["target"].cpu().numpy()
            
            # PM2.5 is target index 0
            y_true_pm25 = targets[:, 0]
            
            # Multimodal
            out = model(spat, temp, tab)
            pred_multi = out["pm25"].cpu().numpy()
            
            # Tabular MLP
            out_tab = tabular_model(tab)
            pred_tab = out_tab[:, 0].cpu().numpy()
            
            all_y_true.extend(y_true_pm25)
            all_pred_multimodal.extend(pred_multi)
            all_pred_tabular.extend(pred_tab)
            
    # Evaluate IDW and Nearest Station on the EXACT same spatial holdout validation records
    for rec in val_records:
        all_pred_idw.append(idw.predict_point(rec["lat"], rec["lon"], "pm25"))
        all_pred_nearest.append(nearest.predict_point(rec["lat"], rec["lon"], "pm25"))
        
    metrics_multimodal = compute_metrics(all_y_true, all_pred_multimodal)
    metrics_tabular = compute_metrics(all_y_true, all_pred_tabular)
    metrics_idw = compute_metrics(all_y_true, all_pred_idw)
    metrics_nearest = compute_metrics(all_y_true, all_pred_nearest)
    
    benchmarks = {
        "target": "PM2.5 Concentration (ug/m3)",
        "evaluation_protocol": "Spatial Holdout Cross-Validation (Leave-Stations-Out: Dwarka, Noida Sec 62, Bawana)",
        "models": {
            "Nearest Station Baseline": {
                "type": "Naive Spatial Heuristic",
                **metrics_nearest,
            },
            "Inverse Distance Weighting (IDW)": {
                "type": "GIS Interpolation Baseline",
                **metrics_idw,
            },
            "Tabular MLP": {
                "type": "Non-spatial ML Regressor",
                **metrics_tabular,
            },
            "Multimodal Deep Learning (Proposed)": {
                "type": "CNN + Bi-LSTM + Cross-Attention + PINN Loss",
                **metrics_multimodal,
            },
        },
    }
    
    benchmark_file = MODELS_DIR / "benchmarks.json"
    with open(benchmark_file, "w", encoding="utf-8") as f:
        json.dump(benchmarks, f, indent=2)
        
    print("\nBenchmark Results on Unseen Spatial Hold-Out Regions:")
    for m_name, m_stats in benchmarks["models"].items():
        print(f"{m_name:38} | RMSE: {m_stats['rmse']:5.2f} | MAE: {m_stats['mae']:5.2f} | R2: {m_stats['r2']:5.3f} | Pearson r: {m_stats['pearson_r']:5.3f}")
        
    return benchmarks


if __name__ == "__main__":
    run_training_pipeline(epochs=10, batch_size=32)
