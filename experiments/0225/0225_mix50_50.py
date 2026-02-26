#!/usr/bin/env python3
from recipe_0225_mixed import main

# Preset: official 50% + manta 50%
if __name__ == "__main__":
    import sys
    sys.argv = [sys.argv[0], "--official-ratio", "50", "--manta-ratio", "50", "--tag", "mix50_50"]
    main()
