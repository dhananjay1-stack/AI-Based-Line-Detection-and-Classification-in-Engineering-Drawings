"""Check how SMP DeepLabV3Plus forward works internally."""
import torch
import segmentation_models_pytorch as smp
import inspect

model = smp.DeepLabV3Plus(encoder_name="resnet50", encoder_weights=None, classes=10)

# Check decoder signature
print("Decoder class:", type(model.decoder).__name__)
print("Decoder forward signature:", inspect.signature(model.decoder.forward))

# Check what encoder returns
x = torch.randn(1, 3, 512, 512)
features = model.encoder(x)
print(f"\nEncoder output type: {type(features)}")
if isinstance(features, (list, tuple)):
    print(f"Number of feature maps: {len(features)}")
    for i, f in enumerate(features):
        print(f"  features[{i}]: {f.shape}")

# Try decoder
try:
    out = model.decoder(*features)
    print(f"\nDecoder output (unpacked): {out.shape}")
except Exception as e:
    print(f"\nDecoder (*features) failed: {e}")
    try:
        out = model.decoder(features)
        print(f"Decoder (features) worked: {out.shape}")
    except Exception as e2:
        print(f"Decoder (features) also failed: {e2}")

# Check the full forward
seg_head_out = model.segmentation_head(out)
print(f"Seg head output: {seg_head_out.shape}")

# Try the model's own forward
full_out = model(x)
print(f"Full forward output: {full_out.shape}")
