# Opensource Clipping — Portal & Compliance Pages

This repository (or branch) contains the static landing portal and official compliance documents for **Opensource Clipping** and **Automation Clippers** (registered in the Meta Developer Console as `Automation_Clppers`).

These files are hosted publicly (typically via **GitHub Pages**) to serve as the required validation endpoints for the Meta Graph API integration.

## 🚀 Purpose

When registering an app on Meta for Facebook Page automation, Meta requires several public URL endpoints:
1. **Privacy Policy URL** (mapped to `privacy-policy.html`)
2. **User Data Deletion Instructions URL** (mapped to `privacy-policy.html#User-data-deletion`)
3. **Terms of Service URL** (mapped to `terms-of-service.html`)

This static portal provides a professional interface that:
- Fulfills Meta's audit requirements for app verification.
- Describes the application's data boundaries, access permissions, and security.
- Features a **Meta App Configuration Portal** with quick "Copy URL" buttons to easily grab and paste these links into the Meta Developer Console.

## 🎨 Tech Stack & Features

- **Styling:** Vanilla Tailwind CSS via CDN.
- **Typography:** Google Fonts (`Outfit` for headings, `Inter` for body).
- **Icons:** Lucide Icons via CDN.
- **Theme:** Default Light Mode (white/slate) with a **Dark Mode Toggle** (dark blue/indigo/violet). Theme preference is persisted in `localStorage`.
- **Responsive Layout:** Designed to work perfectly across mobile, tablet, and desktop viewports.

## 📂 Project Structure

- `index.html`: The main portal landing page containing the compliance copy board and integration console.
- `privacy-policy.html`: The privacy policy containing information about data access, retention, and deletion instructions.
- `terms-of-service.html`: The terms of service outlining usage guidelines and disclaimers.
- `README.md`: Project description and guides.

## 🛠 Deployment to GitHub Pages

To make these URLs active for Meta, deploy the repository to GitHub Pages:
1. Push this branch/repository to GitHub.
2. In the repository settings, navigate to **Pages**.
3. Under **Build and deployment**, select **Deploy from a branch** and select your active branch (e.g. `main` or `gh-pages`) and root folder `/`.
4. Save the configuration. GitHub will provide a public URL like:
   `https://<username>.github.io/<repository-name>/`
5. Copy the generated URLs and paste them into the Meta App Dashboard fields as shown in the developer settings.
