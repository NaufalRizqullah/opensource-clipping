# 🔑 YouTube API Setup Guide

Complete step-by-step guide to set up the YouTube Data API v3 and generate the `youtube_token.json` required for the auto-upload feature.

---

## Overview

The auto-uploader needs an **OAuth 2.0 token** to upload videos to your YouTube channel. Here's the full flow:

```
Google Cloud Console
        ↓
Create Google Cloud Project
        ↓
Enable YouTube Data API v3
        ↓
Configure OAuth Consent Screen
        ↓
Create OAuth Desktop Client ID
        ↓
Download client_secret.json
        ↓
Place in .credentials/client_secret.json
        ↓
Run generate_youtube_token.py (on your local PC)
        ↓
Login Google + Allow YouTube access in browser
        ↓
Output: .credentials/youtube_token.json
        ↓
Test refresh token
        ↓
Use token for auto-upload (local / Colab / Kaggle)
```

---

## Step 1 — Create a Google Cloud Project

1. Go to [Google Cloud Console](https://console.cloud.google.com/)
2. Click the **project dropdown** at the top of the page
3. Click **New Project**
4. Enter a project name (e.g., `My-Clipping-Uploader`)
5. Click **Create** and wait for the project to be provisioned
6. Make sure the newly created project is **selected** in the project dropdown

---

## Step 2 — Enable YouTube Data API v3

1. In the left sidebar, navigate to **APIs & Services → Library**
2. Click **Enable APIs and Services** at the top
3. In the search bar, type **"YouTube"**
4. Scroll down and find **YouTube Data API v3** (not YouTube Analytics or other variants)
5. Click on it, then click the blue **Enable** button
6. Wait for the API to be enabled — you'll be redirected to the API overview page

---

## Step 3 — Configure OAuth Consent Screen

Before creating OAuth credentials, Google requires you to configure a consent screen.

1. In the left sidebar, go to **APIs & Services → OAuth consent screen**
   - On the newer Google Cloud UI, this may appear under **Google Auth Platform → Branding** or **Audience**

2. Select **User type**: `External`, then click **Create**

3. Fill in the required fields:

   | Field | What to enter |
   |---|---|
   | **App name** | `YouTube Auto Uploader` (or any name you prefer) |
   | **User support email** | Your own Google email address |
   | **Developer contact email** | Your own Google email address |

4. Click **Save and Continue**

5. On the **Scopes** page, click **Add or Remove Scopes** and add these two scopes:

   ```
   https://www.googleapis.com/auth/youtube.upload
   https://www.googleapis.com/auth/youtube.readonly
   ```

   > The `youtube.upload` scope is **mandatory** for video uploading. The `youtube.readonly` scope is used for channel verification.

6. Click **Update** → **Save and Continue**

7. On the **Test users** page, click **+ Add Users** and enter the **Google email address** that owns the YouTube channel you want to upload to

8. Click **Add** → **Save and Continue**

9. Review the summary and click **Back to Dashboard**

---

## Step 4 — Create OAuth Client ID

1. In the left sidebar, go to **APIs & Services → Credentials**
2. Click the **+ Create Credentials** button at the top
3. Select **OAuth client ID** from the dropdown
4. For **Application type**, choose: `Desktop app`
5. Enter a name, for example: `YouTube Auto Uploader Desktop`
6. Click **Create**
7. A dialog will appear showing your Client ID and Client Secret — click **Download JSON**
8. The downloaded file will have a long name like:
   ```
   client_secret_123456789-abcdef.apps.googleusercontent.com.json
   ```

> **Important:** This JSON file is your OAuth Client Secret — keep it private and never commit it to a public repository.

---

## Step 5 — Place the Client Secret File

1. **Rename** the downloaded JSON file to:
   ```
   client_secret.json
   ```

2. In your project root, create the `.credentials` folder if it doesn't exist:

   **Linux / macOS:**
   ```bash
   mkdir -p .credentials
   ```

   **Windows PowerShell:**
   ```powershell
   New-Item -ItemType Directory -Force -Path .credentials
   ```

3. Move the renamed file into it:
   ```
   .credentials/client_secret.json
   ```

4. Your project structure should now look like:

   ```text
   opensource-clipping/
   ├── .credentials/
   │   └── client_secret.json       ← Your OAuth Client JSON
   ├── youtube_uploader/
   │   ├── generate_youtube_token.py
   │   └── uploader.py
   └── ...
   ```

---

## Step 6 — Install Dependencies

Make sure the required Google libraries are installed (these are included in `requirements.txt`, but if needed):

```bash
pip install google-auth google-auth-oauthlib google-api-python-client
```

> **Important:** You must generate the token on your **local PC/laptop** — not on Colab or Kaggle — because the OAuth flow opens a browser window for Google login.

---

## Step 7 — Delete Old Token (If Any)

If you previously had a token that expired or was revoked, delete it before generating a new one:

**Linux / macOS:**
```bash
rm -f .credentials/youtube_token.json
```

**Windows PowerShell:**
```powershell
Remove-Item .credentials\youtube_token.json -ErrorAction SilentlyContinue
```

This is important because an expired token with `invalid_grant` status cannot be refreshed — you must re-authorize from scratch.

---

## Step 8 — Generate the Token

Run the token generator from the **project root directory**:

```bash
python youtube_uploader/generate_youtube_token.py
```

> **Path note:** The script looks for `.credentials/client_secret.json` relative to the **current working directory**, so always run from the project root.

### What happens next:

1. **Browser opens automatically** with a Google sign-in page
2. **Select your Google account** — choose the one that owns your YouTube channel
3. **You may see an "unverified app" warning** — since this is your own personal app, it's safe to proceed:
   - Click **Advanced** (or "Show Advanced")
   - Click **Go to [Your App Name] (unsafe)**
   - Click **Continue**
4. **Grant YouTube permissions** — click **Allow** / **Continue** for both `youtube.upload` and `youtube.readonly` scopes
5. **Success!** — The terminal will show a confirmation message and the token file will be created at:
   ```
   .credentials/youtube_token.json
   ```

---

## Step 9 — Verify the Token

### Check the file exists

**Windows:**
```powershell
dir .credentials
```

**Linux / macOS:**
```bash
ls -la .credentials
```

You should see both files:
```
client_secret.json
youtube_token.json
```

### Check the token contents

Open `youtube_token.json` and verify it contains these essential fields:

```json
{
  "token": "ya29.a0AfH...",
  "refresh_token": "1//0eXXXXXX...",
  "token_uri": "https://oauth2.googleapis.com/token",
  "client_id": "123456789-xxxxx.apps.googleusercontent.com",
  "client_secret": "GOCSPX-xxxxx",
  "scopes": [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube.readonly"
  ]
}
```

> **Critical:** The `refresh_token` field **must exist**. Without it, the access token will expire in ~1 hour and cannot be renewed automatically. If it's missing, delete the token and re-run Step 8.

### Test the refresh token

Run the test script included in the project:

```bash
python youtube_uploader/test_youtube_refresh.py
```

**Expected output:**
```
Token valid: True/False
Token expired: True/False
Ada refresh_token: True
Mencoba refresh token...
Refresh OK.
Token valid setelah refresh: True
Channel terdeteksi: Your Channel Name
Channel ID: UCxxxxxxxx
```

If you see `invalid_grant: Token has been expired or revoked`, the token is invalid — go back to Step 7 and re-generate.

---

## Step 10 — Start Uploading!

### Test upload (first video only)

```bash
python scripts/run_upload.py --test-mode
```

### Full upload with scheduling

```bash
python scripts/run_upload.py --interval-hours 12 --tz-name "Asia/Jakarta"
```

The uploader will:
- Read clips and metadata from `outputs/render_manifest.json`
- Upload each video with AI-generated titles, descriptions, and tags
- Schedule uploads at the specified interval
- **Automatically refresh** the access token when it expires (using the `refresh_token`)

---

## Using on Kaggle / Colab

> **Do NOT generate the token on Kaggle or Colab.** The OAuth flow requires a local browser.

1. Generate the token on your **local PC/laptop** (Steps 1-9 above)
2. Upload `.credentials/youtube_token.json` to Kaggle as a secret or private dataset file
3. In your notebook, copy the token to the working directory:

   ```python
   import shutil, os
   os.makedirs("/kaggle/working/.credentials", exist_ok=True)
   shutil.copy("/kaggle/input/your-dataset/youtube_token.json",
               "/kaggle/working/.credentials/youtube_token.json")
   ```

4. Run the uploader as usual:
   ```python
   !python scripts/run_upload.py --test-mode
   ```

---

## Preventing Token Expiration (7-Day Limit)

### The Problem

If your OAuth consent screen is in **Testing** mode, Google imposes a restriction where refresh tokens expire after **7 days**. This applies to all apps with sensitive scopes like `youtube.upload`.

### The Solution: Move to Production

1. Go to **APIs & Services → OAuth consent screen** (or **Google Auth Platform → Audience**)
2. Look for the **Publishing status** — it will say `Testing`
3. Click the **Publish App** button (sometimes labeled "Move to production")
4. Confirm the action
5. The status should change to: `In production`

### After switching to Production:

1. **Delete** your old token:
   ```bash
   rm -f .credentials/youtube_token.json
   ```
2. **Re-run** `generate_youtube_token.py` (Step 8)
3. **Login again** and authorize — the new token will **not** have the 7-day expiration
4. **Test** with `test_youtube_refresh.py` (Step 9)

> **Note:** An "In production" app that hasn't been verified by Google will show a "This app isn't verified" warning during login. For personal use, this is completely safe — just click **Advanced → Go to app → Continue**.

### If "Publish App" button doesn't appear

Your consent screen setup may be incomplete. Go back and make sure all required fields are filled:
- App name
- User support email
- Developer contact email
- At least one scope added
- At least one test user added

After completing the setup, the **Publish App** button should appear.

### Final Checklist

```
✅ OAuth consent screen → User type: External
✅ Publishing status: In production
✅ OAuth Client → Type: Desktop app
✅ youtube_token.json → refresh_token field exists
✅ Token was generated AFTER switching to In production
```

---

## Troubleshooting

| Issue | Solution |
|---|---|
| `invalid_grant: Token has been expired or revoked` | Delete old token → re-generate with `generate_youtube_token.py` |
| `refresh_token` missing in token JSON | Delete token → re-run generator (it uses `access_type="offline"` and `prompt="consent"`) |
| Token expires after 7 days | Change OAuth app from "Testing" → "In production" (see section above) |
| "Google hasn't verified this app" warning | Click **Advanced → Go to app → Continue** (safe for personal use) |
| `client_secret.json not found` | Ensure you're running from the project root where `.credentials/` folder exists |
| Browser doesn't open during token generation | Copy the URL printed in the terminal and paste it into your browser manually |
| `FileNotFoundError` on Kaggle/Colab | Generate the token on your local PC first, then upload it to Kaggle/Colab |
| Channel not detected after token refresh | Make sure you authorized with the Google account that owns the YouTube channel |

---

## Quick Reference Summary

```
 1. Go to console.cloud.google.com → Create new project
 2. APIs & Services → Library → Enable "YouTube Data API v3"
 3. APIs & Services → OAuth consent screen → Configure (External, add scopes & test user)
 4. APIs & Services → Credentials → + Create Credentials → OAuth client ID → Desktop app
 5. Download the JSON file → Rename to client_secret.json
 6. Place at .credentials/client_secret.json
 7. pip install google-auth google-auth-oauthlib google-api-python-client
 8. Delete old .credentials/youtube_token.json (if any)
 9. Run: python youtube_uploader/generate_youtube_token.py
10. Login & authorize YouTube access in browser
11. Verify: python youtube_uploader/test_youtube_refresh.py
12. Upload: python scripts/run_upload.py --test-mode
```

---

## See Also

- [YouTube Auto-Upload](YouTube-Auto-Upload) — Upload commands and scheduling options
- [Getting Started](Getting-Started) — General project setup
- [Google Colab Guide](Google-Colab-Guide) — Running on cloud GPUs
