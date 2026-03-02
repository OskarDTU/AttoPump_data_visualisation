"""Command-line interface."""

import click


@click.command()
@click.version_option()
def main() -> None:
    """atto1."""


if __name__ == "__main__":
    main(prog_name="Attopump_data_visualisation_1")  # pragma: no cover
