#!/usr/bin/env python3
from recipe_0225_mixed import main

# Preset: official 30% + manta 70%
if __name__ == "__main__":
    import sys
    sys.argv = [sys.argv[0], "--official-ratio", "30", "--manta-ratio", "70", "--tag", "mix30_70"]
    main()
