import hashlib
import logging

# Functie om de SHA256-hash van een bestand te berekenen
def bereken_bestand_hash(bestand_pad):
    try:
        sha256_hash = hashlib.sha256()
        with open(bestand_pad, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        logging.error(f"Kon de hash voor bestand {bestand_pad} niet berekenen: {e}")
        return None