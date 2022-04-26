import sys
if sys.prefix == sys.base_prefix:
    print("No, you are not in a virtual environment.")
else:
    print("Yes, you are in a virtual environment.")