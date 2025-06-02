import sys
import os

# Tambahkan path folder aplikasi ke sys.path agar bisa import app.py
sys.path.insert(0, os.path.dirname(__file__))

from app import app as application
