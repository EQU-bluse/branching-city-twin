# Branching City Twin

An event-driven backend for reconstructing city state at a historical point, creating simulation branches, and tracing the effects of operational decisions.

## Requirements

- Python 3.12+

## Public commands

Run `python -m city_twin status` to inspect the backend identity and readiness. Run the baseline tests with `python -m unittest discover -s tests -v`.

The initial baseline intentionally contains only the runnable service shell. Domain capabilities are added through reviewed backend tasks.
