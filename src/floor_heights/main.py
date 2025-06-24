"""Entry point for the floor-heights package."""

import os
import sys

from dotenv import find_dotenv, load_dotenv


def main():
    """Main entry point for the command line interface."""

    dotenv_path = find_dotenv(filename=".env", raise_error_if_not_found=True)
    load_dotenv(dotenv_path, override=True)

    if not os.getenv("DB_CONNECTION_STRING"):
        print("ERROR: DB_CONNECTION_STRING environment variable not found")
        print("Make sure you have a .env file with this variable in the project root")
        return 1

    print("Floor Heights Pipeline")
    print("=====================")
    print("For region-specific pipeline execution, use the individual stage modules.")
    print("Example: python -m floor_heights.pipeline.stage01_import_and_process_building_levels --region wagga")
    return 0


if __name__ == "__main__":
    sys.exit(main())
