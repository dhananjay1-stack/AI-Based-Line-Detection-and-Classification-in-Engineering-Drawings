import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Lazy imports — avoid circular / heavy imports at package init
__all__ = ['LineDetectionInference']
