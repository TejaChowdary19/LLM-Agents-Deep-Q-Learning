from autorom.accept_rom_license import accept_license_and_download
import os

print("Accepting ROM license and downloading ROMs...")
rom_directory = os.path.join(os.getcwd(), "roms")
os.makedirs(rom_directory, exist_ok=True)
accept_license_and_download(rom_directory)
print(f"ROMs downloaded to: {rom_directory}")