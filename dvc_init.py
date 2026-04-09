import os
import subprocess
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def run_cmd(cmd):
    log.info(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        log.error(f"Error: {result.stderr}")
    else:
        log.info(result.stdout)
    return result.returncode

def main():
    # 1. Initialize DVC
    if not os.path.exists(".dvc"):
        log.info("Initializing DVC...")
        run_cmd("dvc init --no-scm") # Use --no-scm if git is not initialized or handled separately
    else:
        log.info("DVC already initialized.")

    # 2. Track Data Directories
    dirs_to_track = [
        "datasets",
        "outputs/processed_data",
        "outputs/processed_data_hf"
    ]

    for d in dirs_to_track:
        if os.path.exists(d):
            log.info(f"Tracking directory: {d}")
            run_cmd(f"dvc add {d}")
        else:
            log.warning(f"Directory {d} not found. Skipping.")

    log.info("\n[DVC Setup Complete]")
    log.info("Next steps:")
    log.info("1. Configure a remote: dvc remote add -d myremote s3://mybucket/stress_project")
    log.info("2. Push data: dvc push")

if __name__ == "__main__":
    main()
