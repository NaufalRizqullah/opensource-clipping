"""
generate_youtube_token.py

Purpose of this file:
- Open Google login in the browser.
- Request the YouTube access scopes the uploader needs.
- Create the .credentials/youtube_token.json file.
- This token file is later used by run_upload.py / uploader.py to upload videos to YouTube.

Run this file only the first time you create the token,
or when the old token is broken / you want to re-login to the YouTube account.
"""

import os

# This library runs the OAuth flow for a desktop/local app.
# When run, the browser opens for the Google login.
from google_auth_oauthlib.flow import InstalledAppFlow


# Scope = the access permissions requested from the Google/YouTube account.
# youtube.upload   -> permission to upload videos to YouTube.
# youtube.readonly -> permission to read channel/video data, e.g. to check the existing upload schedule.
YOUTUBE_SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly",
]


# The OAuth Client ID file from the Google Cloud Console.
# This file is obtained from:
# Google Cloud → APIs & Services → Credentials → OAuth client ID → Desktop app → Download JSON
#
# Rename the downloaded file to client_secret.json
# and save it in the .credentials/ folder.
CLIENT_SECRET_FILE = ".credentials/client_secret.json"


# The output token file produced by the Google login.
# This is the file that uploader.py reads later.
TOKEN_FILE = ".credentials/youtube_token.json"


def main():
    """
    Main function to generate the YouTube token.

    Flow:
    1. Check whether client_secret.json exists.
    2. Create the .credentials folder if it doesn't exist.
    3. Open the OAuth login in the browser.
    4. After the user approves, save the token to youtube_token.json.
    """

    # Check whether the client_secret.json file is available.
    # If it doesn't exist, the script stops because the OAuth flow cannot start.
    if not os.path.exists(CLIENT_SECRET_FILE):
        raise FileNotFoundError(
            f"{CLIENT_SECRET_FILE} not found. "
            "Download the OAuth Client ID JSON from Google Cloud, rename it to client_secret.json, "
            "then save it into the .credentials/ folder"
        )

    # Make sure the .credentials folder is available.
    # If it doesn't exist, it is created automatically.
    os.makedirs(".credentials", exist_ok=True)

    # Build the OAuth flow from the client_secret.json file.
    # Here we pass the list of required YouTube scopes/permissions.
    flow = InstalledAppFlow.from_client_secrets_file(
        CLIENT_SECRET_FILE,
        scopes=YOUTUBE_SCOPES,
    )

    # Run the OAuth login via a local browser.
    #
    # port=0:
    #   Python picks a free port automatically.
    #
    # access_type="offline":
    #   Requests a refresh_token so the token can be refreshed automatically
    #   without logging in again every time the access_token expires.
    #
    # prompt="consent":
    #   Forces Google to show the consent screen again.
    #   This is useful so the refresh_token is actually returned.
    creds = flow.run_local_server(
        port=0,
        access_type="offline",
        prompt="consent",
    )

    # Save the resulting credential/token to the youtube_token.json file.
    # This file contains access_token, refresh_token, client_id, client_secret, and scope.
    #
    # IMPORTANT:
    # Do not upload youtube_token.json to GitHub.
    # Treat this file like your YouTube access password.
    with open(TOKEN_FILE, "w", encoding="utf-8") as f:
        f.write(creds.to_json())

    print(f"✅ Token created successfully: {TOKEN_FILE}")


# This makes main() run only when this file is executed directly:
#
# python generate_youtube_token.py
#
# If this file is imported by another Python file, main() does not run automatically.
if __name__ == "__main__":
    main()
