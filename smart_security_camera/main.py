# main.py
"""
Smart Security Camera System — CSY3058 Media Technology
Entry point. Launches the Tkinter dashboard.

Author: Liza Gurung 
Student ID: 24812928
Date: May 2026
"""

import tkinter as tk
from ui.dashboard import Dashboard


def main():
    root = tk.Tk()
    app  = Dashboard(root)
    root.protocol("WM_DELETE_WINDOW", lambda: (app._stop(), root.destroy()))
    root.mainloop()


if __name__ == "__main__":
    main()