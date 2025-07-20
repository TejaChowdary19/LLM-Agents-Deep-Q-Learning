# Save this as install_roms.py
try:
    import autorom
    print("AutoROM is installed.")
    try:
        from autorom import accept_rom_license
        print("ROM license module found.")
        try:
            accept_rom_license.install_roms(".")
            print("ROMs installed successfully.")
        except Exception as e:
            print(f"Error installing ROMs: {e}")
    except ImportError:
        print("ROM license module not found.")
except ImportError:
    print("AutoROM is not installed. Installing now...")
    import pip
    pip.main(['install', 'autorom', 'autorom[accept-rom-license]'])
    print("Please run this script again after installation.")