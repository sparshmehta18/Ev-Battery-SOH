import sys,traceback
try:
    import pandas, numpy, scipy
    print('pandas', pandas.__version__)
    print('numpy', numpy.__version__)
    print('scipy', scipy.__version__)
except Exception:
    traceback.print_exc()
    sys.exit(1)
