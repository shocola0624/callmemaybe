import sys


def _entry_point() -> None:
    """Run the main program with error handling."""
    try:
        from .call_me_maybe import call_me_maybe

        call_me_maybe()

    except ImportError as e:
        print(
            "Error: failed to import required modules.\n"
            f"Detail: {e}\n"
            "Hint: run 'uv sync' or 'make install' to install dependencies.",
            file=sys.stderr
        )
        sys.exit(1)

    except KeyboardInterrupt:
        print(
            "\nThe process has been successfully interrupted.\n",
            file=sys.stdout
        )
        sys.exit(0)


if __name__ == "__main__":
    _entry_point()
