import argparse
import json

from city_twin import __version__


def main() -> int:
    parser = argparse.ArgumentParser(prog="city_twin")
    parser.add_argument("command", choices=["status"])
    args = parser.parse_args()
    if args.command == "status":
        print(json.dumps({"service": "branching-city-twin", "status": "ready", "version": __version__}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
