# Python Coding Style Guide & Best Practices

## 1. Guiding Principles

The primary goal is to produce code that is **clear, robust, maintainable, and Pythonic**.

* **Clarity over Cleverness**: Write code that is easy to understand. Avoid obscure language features.
* **Simplicity First**: Prefer straightforward solutions over unnecessary complexity.
* **Pythonic Code**: Leverage Python's idioms and standard library effectively.
* **Maintainability**: Structure code to be modular, facilitating future changes and extensions.
* **Self-Documenting**: Strive for descriptive names and a logical structure that minimizes the need for explanatory comments.
* **Robustness**: Anticipate and handle potential errors gracefully.

---

## 2. Assistant Behavior
* If I tell you that you are wrong, re-evaluate your response based on the facts provided.
* Avoid apologies or conciliatory statements like "You're right" or "Yes."
* Stick to the task at hand. Avoid hyperbole and unnecessary conversational filler.
* Ensure responses are relevant to the provided context and code. Keep them concise.
* Think step-by-step to validate your reasoning before responding.
* For each code snippet, provide a brief explanation of the changes made and why they improve the code.
* When refactoring, ensure existing functionality is preserved unless explicitly instructed otherwise.
* When correcting errors, provide a clear explanation of the issue and how the fix addresses it.
* When suggesting libraries or tools, prefer widely adopted and well-maintained options.
* When working with data structures, prefer built-in types and standard library collections unless a third-party library offers significant advantages.


---

## 3. Style & Formatting

#### Naming Conventions
* **Variables & Functions**: `snake_case`.
* **Constants**: `UPPER_SNAKE_CASE`.
* **Classes**: `PascalCase`.
* **Abbreviations**: Domain-specific abbreviations (e.g., `img`, `rso`, `cfg`) are acceptable if they are unambiguous and enhance readability.

#### Code Structure
* **Indentation**: 4 spaces per indentation level.
* **Line Length**: Keep lines under 120 characters.
* **Nesting**: Avoid deeply nested logic. Decompose complex blocks into smaller, more focused functions.
* **Imports**: Group imports in the following order, with each group sorted alphabetically:
    1.  Standard library
    2.  Third-party packages
    3.  Local application/library specific imports
* **Section Separators**: Use comments to break up large files into logical sections for better readability.
    ```python
    # ==============================================================================
    # Constants
    # ==============================================================================
    ```

---

## 4. Documentation & Comments

#### Docstrings
* All public modules, classes, functions, and methods must have a docstring that follows the **Google Python Style Guide**.
* Use triple double-quotes (`"""Docstring content..."""`). The summary line should be on the same line as the opening quotes.
* Include `Args:`, `Returns:`, `Yields:`, and `Raises:` sections where applicable.
* For complex functions, consider adding an `Example:` or `Notes:` section.

#### Inline Comments
* Use comments sparingly, only to clarify non-obvious logic.
* Employ the "Better Comments" style for annotations to improve code scanning:
    ```python
    # * Highlights very important information.
    # ? Asks a question or indicates a clarification is needed.
    # ! Warns about potential issues or gotchas.
    # TODO: Marks a task that needs to be completed.
    ```

---

## 5. Architecture & Design

* **Architecture Decisions**: For significant design choices (e.g., selecting a library, designing a core algorithm), add a brief document in a `docs/` directory. This provides context on *why* a decision was made.
* **Diagrams**: Use `mermaid` syntax in Markdown files (`.md`) to visualize architecture, workflows, or data flows.


---

## 6. Project Structure

### The Src Layout (Recommended for Robustness)

For any project intended for installation or distribution, the `src` layout is highly recommended. It provides a clean separation between your actual source code and repository-level files (e.g., `pyproject.toml`, `tests/`, `README.md`).

* **Prevents Import Conflicts**: It forces you to install the project in editable mode (`pip install -e .`) for development. This ensures that your code runs from the installed package, preventing accidental imports of local files from the project root.
* **Clear Separation**: Your importable package (`image_processor`) is cleanly nested within `src/`, making the project structure unambiguous.
* **Industry Standard**: This layout is a standard convention for many well-regarded Python libraries and applications.

#### Example Src Structure

```text
image_processor/
│
├── .gitignore
├── pyproject.toml
├── README.md
│
├── conf/
│   └── config.yaml
│
├── src/
│   └── image_processor/
│       ├── __init__.py
│       ├── main.py          # Main entrypoint, run via `python -m image_processor.main`
│       ├── processing.py    # Core application logic
│       └── utils.py         # Utility functions
│
└── tests/
    ├── __init__.py
    └── test_processing.py
```
---

## 7. Code Patterns & Language Features

#### Type Hinting
* All function and method signatures must include type hints.
* Use standard types (`int`, `str`, `list`) and types from the `typing` module (`Optional`, `Iterator`, `Callable`) where necessary.

#### Error Handling
* Handle exceptions specifically (e.g., `try...except FileNotFoundError`).
* Avoid catching generic `Exception` unless it's a last resort and is re-raised or logged with full context.
* Do not use `try...except` blocks for regular control flow.

#### Recommended Language Features
* **File Paths**: Use the `pathlib` module for all filesystem path manipulations.
* **Configuration**: Avoid hardcoded magic strings and numbers. Define them as constants or manage them in configuration files.
* **String Formatting**: Use f-strings (`f"..."`) exclusively.
* **Generators**: Use `yield` for returning sequences, especially in I/O-bound or memory-sensitive operations.

---

## 8. Recommended Libraries

* **Numerical Operations**: `numpy`
* **Data Structures**: `collections` (e.g., `defaultdict`, `Counter`), `dataclasses`
* **Iteration**: `itertools` for efficient looping.
* **File Paths**: `pathlib`
* **CLI Applications**: `typer`
* **Console UI**: `rich` (for tables, progress bars, etc.)
* **Logging**: `loguru`
* **Configuration**: `hydra` / `omegaconf`

---

## 9. Example Code Snippets

### Logging Utility (`src/image_processor/utils.py`)
```python
# ==============================================================================
# Imports
# ==============================================================================
import sys
from pathlib import Path

from loguru import logger
from rich.logging import RichHandler


# ==============================================================================
# Functions
# ==============================================================================
def setup_logger(log_file: Path) -> None:
    """Configures Loguru for rich console logging and persistent file logging.

    Args:
        log_file: The file path where logs will be stored.
    """
    # * Ensure the directory for the log file exists.
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger.remove()  # Remove default handler to prevent duplicate output.

    # Console logger with rich formatting for readability.
    logger.add(
        sys.stderr,
        level="INFO",
        format="{message}",
        enqueue=True,  # Makes logging thread/process-safe.
        backtrace=True,  # Provides full stack trace on exceptions.
        colorize=True,
        handler=RichHandler(rich_tracebacks=True, show_path=False, markup=True),
    )

    # File logger for detailed, persistent records.
    logger.add(
        log_file,
        level="DEBUG",
        rotation="10 MB",
        retention="10 days",
        enqueue=True,
        backtrace=True,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    )

    logger.info("[bold green]Logger configured successfully.[/bold green]")
```

### Core Processing Logic (`src/image_processor/processing.py`)

```python
# ==============================================================================
# Imports
# ==============================================================================
import csv
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

from loguru import logger
from omegaconf import DictConfig
from rich.progress import Progress

# ==============================================================================
# Constants
# ==============================================================================
SUCCESS_STATUS = "SUCCESS"
FAILURE_STATUS = "FAILURE"


# ==============================================================================
# Dataclasses
# ==============================================================================
@dataclass(frozen=True)
class ImageMetadata:
    """Immutable data structure for image metadata."""
    image_id: str
    file_path: Path
    confidence_score: float
    rso_class: str


@dataclass
class ProcessingResult:
    """Mutable data structure for capturing the result of an operation."""
    metadata: ImageMetadata
    status: str
    message: str = ""


# ==============================================================================
# Processor Class
# ==============================================================================
class ImageProcessor:
    """Orchestrates the loading, filtering, and processing of image assets."""

    def __init__(self, cfg: DictConfig):
        """Initializes the processor with a Hydra configuration object.

        Args:
            cfg: The OmegaConf DictConfig object provided by Hydra.
        """
        self.cfg = cfg
        self.input_csv = Path(self.cfg.paths.input_csv)
        self.output_dir = Path(self.cfg.paths.output_dir)
        # * Create the output directory idempotently.
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory set to: '{self.output_dir.resolve()}'")

    def _load_and_filter_metadata(self) -> Iterator[ImageMetadata]:
        """Loads and yields ImageMetadata records meeting filter criteria.

        Uses a generator to remain memory-efficient on large CSV files.

        Yields:
            An iterator of ImageMetadata objects that match the filters.

        Raises:
            FileNotFoundError: If the input_csv path does not exist.
        """
        logger.info(f"Loading metadata from: '{self.input_csv}'")
        if not self.input_csv.is_file():
            raise FileNotFoundError(f"Input CSV not found at: {self.input_csv}")

        try:
            with self.input_csv.open(mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        confidence = float(row["confidence_score"])
                        if confidence < self.cfg.processing.min_confidence:
                            continue

                        required_class = self.cfg.processing.required_class
                        if required_class and row["rso_class"] != required_class:
                            continue

                        yield ImageMetadata(
                            image_id=row["image_id"],
                            file_path=Path(row["file_path"]),
                            confidence_score=confidence,
                            rso_class=row["rso_class"],
                        )
                    except (KeyError, ValueError) as e:
                        logger.warning(f"Skipping malformed row: {row}. Reason: {e}")
                        continue
        except Exception as e:
            logger.opt(exception=True).error("A critical error occurred while reading the CSV.")
            raise e

    def _simulate_image_processing(self, metadata: ImageMetadata) -> ProcessingResult:
        """Simulates a single I/O-bound image processing task.

        Args:
            metadata: The metadata object for the image to be processed.

        Returns:
            A ProcessingResult indicating the outcome.
        """
        try:
            logger.debug(f"Processing image_id: {metadata.image_id}")
            # Simulate I/O work (e.g., file copy, network download).
            time.sleep(self.cfg.processing.simulate_io_delay_seconds)

            if not metadata.file_path.name:  # Trivial failure condition
                raise ValueError("File path appears to be a directory.")

            return ProcessingResult(metadata=metadata, status=SUCCESS_STATUS)
        except Exception as e:
            logger.error(f"Failed to process {metadata.image_id}: {e}")
            return ProcessingResult(metadata=metadata, status=FAILURE_STATUS, message=str(e))

    def run_pipeline(self) -> Tuple[list[ProcessingResult], Counter]:
        """Executes the full processing pipeline.

        Returns:
            A tuple containing a list of all results and a Counter
            summarizing the status of processed RSO classes.
        """
        logger.info("Starting image processing pipeline...")
        metadata_records = list(self._load_and_filter_metadata())

        if not metadata_records:
            logger.warning("No metadata records found matching criteria. Exiting.")
            return [], Counter()

        results: list[ProcessingResult] = []
        class_summary = Counter()

        with Progress() as progress:
            task = progress.add_task("[cyan]Processing...", total=len(metadata_records))
            # * Use ThreadPoolExecutor for I/O-bound tasks to avoid GIL limitations.
            with ThreadPoolExecutor(max_workers=self.cfg.processing.max_workers) as executor:
                future_to_metadata = {
                    executor.submit(self._simulate_image_processing, record): record
                    for record in metadata_records
                }

                for future in as_completed(future_to_metadata):
                    result = future.result()
                    results.append(result)
                    if result.status == SUCCESS_STATUS:
                        class_summary[result.metadata.rso_class] += 1
                    progress.update(task, advance=1)

        logger.info("Image processing pipeline finished.")
        return results, class_summary
```

### Main Entrypoint (`src/image_processor/main.py`)

```python
# ==============================================================================
# Imports
# ==============================================================================
from collections import Counter
from pathlib import Path

import hydra
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from rich.console import Console
from rich.table import Table

# Local application imports (using relative imports)
from .processing import ImageProcessor, ProcessingResult, SUCCESS_STATUS
from .utils import setup_logger

# ==============================================================================
# Functions
# ==============================================================================
def generate_summary_table(
    results: list[ProcessingResult], class_summary: Counter
) -> Table:
    """Creates a rich Table to display the processing summary.

    Args:
        results: A list of all processing results from the pipeline.
        class_summary: A Counter of successfully processed RSO classes.

    Returns:
        A rich Table object ready for printing.
    """
    table = Table(title="[bold blue]Image Processing Pipeline Summary[/bold blue]")
    table.add_column("Metric", style="dim", width=30)
    table.add_column("Value", justify="right")

    success_count = sum(1 for r in results if r.status == SUCCESS_STATUS)
    failure_count = len(results) - success_count

    table.add_row("Total Images Considered", str(len(results)))
    table.add_row("[green]Successful Operations[/green]", str(success_count))
    table.add_row("[red]Failed Operations[/red]", str(failure_count))
    table.add_section()

    if class_summary:
        table.add_row("[bold]Success Count by RSO Class[/bold]", "")
        for rso_class, count in sorted(class_summary.items()):
            table.add_row(f"  - {rso_class}", str(count))

    return table


# The config_path is relative to this file's location.
# With main.py in src/image_processor, ../../conf points to the root `conf` dir.
@hydra.main(config_path="../../conf", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    """Main function orchestrated by Hydra.

    Args:
        cfg: The configuration object populated by Hydra.
    """
    console = Console()
    setup_logger(Path(cfg.log_file))
    logger.info(f"Starting [bold cyan]{cfg.app_name}[/bold cyan]...")
    logger.debug(f"Full configuration:\n{OmegaConf.to_yaml(cfg)}")

    try:
        processor = ImageProcessor(cfg)
        results, class_summary = processor.run_pipeline()
    except Exception:
        logger.opt(exception=True).critical("Pipeline execution failed.")
        console.print("[bold red]A critical error occurred. Check logs for details.[/bold red]")
        return  # Exit gracefully

    if not results:
        logger.info("No results to report.")
        console.print("[yellow]Pipeline ran but produced no results to summarize.[/yellow]")
        return

    summary_table = generate_summary_table(results, class_summary)
    console.print(summary_table)
    logger.info("Summary report displayed.")


if __name__ == "__main__":
    main()
```
