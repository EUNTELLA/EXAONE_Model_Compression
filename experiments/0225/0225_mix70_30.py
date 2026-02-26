#!/usr/bin/env python3
from recipe_0225_mixed import main

# Preset: official 70% + manta 30%
if __name__ == "__main__":
    import sys
    sys.argv = [sys.argv[0], "--official-ratio", "70", "--manta-ratio", "30", "--tag", "mix70_30"]
    main()
