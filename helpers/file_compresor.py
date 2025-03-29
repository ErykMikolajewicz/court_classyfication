from pathlib import Path
import gzip


def compress_directory(path: Path):

    for file_path in path.rglob('*'):
        if file_path.is_dir():
            continue

        gz_file_path = file_path.with_suffix(file_path.suffix + '.gz')

        with file_path.open('rb') as f_in:
            data_to_compress = f_in.read()

        with gzip.open(gz_file_path, 'wb') as f_out:
            f_out.write(data_to_compress)

        print(f"Compressed: {file_path}")


def delete_not_compressed(path: Path):

    for file_path in path.rglob('*'):
        if file_path.is_dir():
            continue
        file_path: Path

        if file_path.suffix == '':
            file_path.unlink()
            print(f'Deleted: {file_path}')


if __name__ == "__main__":
    target_directory = Path("../data/raw")
    compress_directory(target_directory)
    delete_not_compressed(target_directory)