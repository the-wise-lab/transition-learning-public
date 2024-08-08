import requests
import zipfile
import io
import os
import sys

# URL of the zip file (hardcoded)
zip_url = "https://figshare.com/ndownloader/files/48178787?private_link=695c37a72194f4d9cd58"


def download_and_unzip(url, dest_folder="."):
    """
    Download a zip file from a URL and unzip it in the destination
    folder with progress indication in GB.
    """
    # Send a GET request to the URL
    with requests.get(url, stream=True) as response:
        response.raise_for_status()  # Check if the download was successful

        # Get the total size of the file in bytes
        total_size = int(response.headers.get("content-length", 0))
        total_gb = total_size / (1024**3)

        print(f"Downloading... Total size: {total_gb:.2f} GB")

        downloaded_size = 0
        with open("downloaded_file.zip", "wb") as file:
            for chunk in response.iter_content(chunk_size=4096):
                if chunk:  # Filter out keep-alive new chunks
                    file.write(chunk)
                    downloaded_size += len(chunk)
                    progress_gb = downloaded_size / (1024**3)
                    print(
                        f"\rDownloaded: {progress_gb:.2f} GB of {total_gb:.2f} GB",
                        end="",
                    )

        print("\nDownload completed. Extracting files...")
        with zipfile.ZipFile("downloaded_file.zip") as thezip:
            thezip.extractall(path=dest_folder)
            print("Extraction completed.")

        print("Cleaning up...")
        os.remove("downloaded_file.zip")

        print("Done")


if __name__ == "__main__":
    # Download and unzip the file to the current directory
    download_and_unzip(zip_url)
