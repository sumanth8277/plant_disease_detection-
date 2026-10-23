# AGROTECH LIVE DEMONSTRATION SCRIPT & EXAMINER WALKTHROUGH

## Project: Plant Disease Detection and Smart Pesticide & Fertilizer Recommendation System
**Interface Name:** Agrotech  
**Target Duration:** 5 to 7 Minutes  
**Audience:** College Project Evaluation Committee / External Examiners

---

## 1. PRE-DEMO CHECKLIST (BEFORE CALLING EXAMINERS)

Before inviting the evaluation panel to your screen:
1. **Activate Virtual Environment:**
   ```powershell
   .\venv\Scripts\Activate.ps1
   ```
2. **Ensure MySQL Server is Running (Optional / Recommended):**
   ```powershell
   Get-Service -Name MySQL* # Verify running status
   ```
3. **Start the Flask Application:**
   ```powershell
   python app.py
   ```
   *Verify terminal shows:* `Running on http://127.0.0.1:5000`
4. **Open Browser:** Navigate to `http://127.0.0.1:5000/` and ensure the landing page displays cleanly.
5. **Prepare Test Sample Images:** Have 2 sample leaf images (e.g., Tomato Bacterial Spot, Potato Early Blight) and 1 non-image text/pdf file ready on your desktop for quick drag-and-drop.
6. **Have a Second Browser Window Open (Incognito/Private):** Useful for demonstrating session isolation and IDOR 403 authorization defense.

---

## 2. TIMED STEP-BY-STEP DEMO SCRIPT

### STEP 1: HOMEPAGE & BRANDING OVERVIEW (0:00 – 0:30)
- **Action:** Point the examiner to the browser at `http://127.0.0.1:5000/`.
- **Script to Speak:**
  > "Good morning examiners. This is **Agrotech: Plant Disease Detection and Smart Pesticide & Fertilizer Recommendation System**. The platform is designed to assist farmers and agronomists in detecting foliar crop diseases early and obtaining exact chemical, organic, and fertilizer treatment schedules."
- **Point out on Screen:**
  - The clean navbar showing the brand **Agrotech** and navigation links (`Home`, `Diagnose`, `Login`, `Register`).
  - The hero section explaining the three-step workflow: Upload Leaf $\rightarrow$ Deep Learning Classification $\rightarrow$ Smart Recommendations.

---

### STEP 2: USER REGISTRATION & INPUT VALIDATION (0:30 – 1:15)
- **Action:** Click on the **Register** link in the navigation bar (`/register`).
- **Script to Speak:**
  > "To provide personalized farm management, Agrotech supports user accounts with secure authentication. Let me demonstrate our server-side input validation."
- **Actions to Show:**
  1. Try submitting the form with an invalid email (e.g., `testuser` without `@`) or a weak password (`123`).
  2. Show that the form blocks the submission and provides immediate, user-friendly validation feedback.
  3. Now register a valid account:
     - **Username:** `farmer_demo`
     - **Email:** `demo@agrotech.local`
     - **Password:** `Farmer@123`
     - **Confirm Password:** `Farmer@123`
  4. Click **Register**. The system flashes `Registration successful! Please log in.` and redirects to `/login`.
- **Script to Speak:**
  > "The password is cryptographically salted and hashed using Werkzeug’s scrypt algorithm before being saved to MySQL. Plaintext passwords are never stored."

---

### STEP 3: LOGIN & SECURE SESSION INITIALIZATION (1:15 – 1:45)
- **Action:** Enter the newly registered credentials:
  - **Username:** `farmer_demo`
  - **Password:** `Farmer@123`
- **Click:** **Log In**.
- **Point out on Screen:**
  - Notice the dynamic navbar transformation: `Login` and `Register` disappear.
  - A personalized badge appears: `Welcome, farmer_demo!`, along with a link to `History` and a secure `Logout` button.
- **Script to Speak:**
  > "Notice how our navbar dynamically updates. Session cookies are secured with HTTPOnly and SameSite flags to prevent XSS session hijacking and CSRF attacks. All forms embed cryptographic CSRF tokens managed by Flask-WTF."

---

### STEP 4: LEAF IMAGE UPLOAD & PREPROCESSING (1:45 – 2:45)
- **Action:** Click **Diagnose / Upload** in the navigation bar (`/upload`).
- **Actions to Show:**
  1. Drag and drop a sample diseased leaf image into the upload dropzone.
  2. Point out the instant client-side image preview and file size validation.
  3. Click **Analyze Leaf Health**.
  4. Point out the animated loading spinner providing real-time feedback while the inference pipeline processes.
- **Script to Speak:**
  > "When the image is submitted, our backend pipeline performs strict validation: verifying the MIME type, sanitizing the filename to prevent directory traversal, converting the image to 3-channel RGB, resizing it to $224 \times 224$, and scaling pixel values to $[-1.0, 1.0]$. The preprocessed tensor is then passed into our MobileNetV2 convolutional neural network."

---

### STEP 5: DIAGNOSTIC RESULTS & SMART RECOMMENDATIONS (2:45 – 3:45)
- **Action:** The browser loads the result page (`/result/<prediction_id>`).
- **Point out on Screen:**
  1. **Detection Header:** Displays the detected disease (e.g., *Tomato - Bacterial Spot*) and confidence level with a visual progress meter.
  2. **Pathogen Profile:** Explains the biological cause (e.g., *Xanthomonas campestris*) and visual foliar symptoms.
  3. **Smart Recommendations Section:**
     - **Chemical Pesticide:** Displays the targeted chemical (e.g., *Copper Hydroxide / Streptomycin*), specific dosage (e.g., *2.5 g per liter of water*), and pre-harvest safety interval.
     - **Organic IPM Alternative:** Displays eco-friendly treatments (e.g., *Neem oil spray, compost tea*) to minimize environmental toxicity.
     - **Fertilizer Guidance:** Advises on balanced Nitrogen vs. Potassium application to strengthen plant resistance.
- **Script to Speak:**
  > "Unlike traditional projects that only output a disease label, Agrotech queries our normalized MySQL database to output actionable treatment schedules following Integrated Pest Management principles. The farmer gets exact dilution rates, application frequencies, and safety waiting periods before harvesting."

---

### STEP 6: PREDICTION HISTORY & OWNERSHIP ISOLATION (3:45 – 4:30)
- **Action:** Click on the **History** link in the navigation bar (`/history`).
- **Point out on Screen:**
  1. The newly performed scan is now listed in the history table with thumbnail, detected disease, confidence badge, and timestamp.
  2. Click **View Details** to open the full diagnostic report.
- **Showcase IDOR Security Defense (Examiners love this!):**
  1. Copy the current result URL (e.g., `http://127.0.0.1:5000/result/1`).
  2. Open an **Incognito / Private Window** (or log in as a different user).
  3. Paste the URL into the incognito window.
  4. Point out that the server returns **HTTP 403 Forbidden / Access Denied**.
- **Script to Speak:**
  > "To satisfy modern cybersecurity standards, we implemented strict ownership validation. If another user attempts an Insecure Direct Object Reference (IDOR) attack to view or scrape someone else's agricultural history, our backend checks user identity against record ownership and immediately blocks access with an HTTP 403 Forbidden response."

---

### STEP 7: SECURITY & EDGE-CASE HANDLING (4:30 – 5:15)
- **Action:** Switch back to the main browser window.
- **Actions to Show:**
  1. Navigate to `/upload` and attempt to upload a `.txt` or `.exe` file.
  2. Show that the upload validator immediately rejects the file with an informative error message: `Invalid file type. Allowed formats: PNG, JPG, JPEG`.
  3. Click **Logout**. Point out that the session is cleared, the navbar resets to guest mode, and logout is strictly executed via an HTTP POST request to prevent logout CSRF.
- **Script to Speak:**
  > "The application handles edge cases defensively. Malicious file extensions and corrupt headers are caught before processing. And logout requires an authenticated POST request with CSRF verification to prevent unauthorized forced logouts."

---

### STEP 8: AUTOMATED TEST SUITE PRESENTATION (5:15 – 5:45)
- **Action:** Switch to the terminal window and execute:
  ```powershell
  .\venv\Scripts\pytest
  ```
- **Point out on Screen:**
  - The rapid execution of all **102 tests** passing in ~27 seconds!
  - Modules passing: `test_auth_service`, `test_auth_routes`, `test_authorization`, `test_predictions`, `test_phase4_routes`, `test_recommendations`, `test_model_pipeline`, `test_image_preprocessing`.
- **Script to Speak:**
  > "To ensure continuous quality and regression prevention, we developed a 102-test automated pytest suite. It covers unit testing, integration pipelines, CSRF protection, and authorization controls, maintaining a 100% pass rate."

---

### STEP 9: CONCLUSION & VIVA INVITATION (5:45 – 6:00)
- **Script to Speak:**
  > "In summary, Agrotech combines high-accuracy deep learning with practical, actionable agronomic knowledge in a secure, production-grade web application. Thank you, examiners. We welcome your questions."

---

## 3. EXAMINER TRAP / SURPRISE SCENARIOS & FALLBACK PLANS

| Surprise Scenario by Examiner | Expected Behavior / Answer | How to Demonstrate Live |
| :--- | :--- | :--- |
| **"What if the MySQL database goes down?"** | The application incorporates defensive fallback logic in `RecommendationService`. It will not throw a 500 server crash; it gracefully displays general disease management precautions and logs an informative warning. | You can point examiners to `services/recommendation_service.py` (`_build_fallback_recommendations`). |
| **"What if the model file is missing or corrupted?"** | `ModelLoader` catches `OSError` and missing weight files, initializing a safe prediction simulator so examiners can still test UI flows and database tracking during reviews. | Point examiners to `services/model_loader.py` fallback handling. |
| **"Can a guest farmer use the tool without registering?"** | **Yes!** Guest scans are fully supported. The prediction is saved with `user_id = NULL`. The guest receives full recommendations, but the scan is simply not attached to a personal user history table. | Open `/upload` while logged out, upload a leaf, and demonstrate that prediction works seamlessly. |
| **"Why is confidence sometimes low?"** | If the uploaded image is blurry, contains extreme glare, or is not a plant leaf, confidence drops. Our pipeline flags confidence under 60% as 'Uncertain' and advises the farmer to retake the photo. | Point to `services/prediction_service.py` threshold logic. |
| **"How do you prevent Cross-Site Scripting (XSS)?"** | Jinja2 templating auto-escapes all dynamic variables by default (`{{ variable }}`). Additionally, `HTTPOnly` session cookies prevent injected JavaScript from reading session cookies. | Point to `config.py` cookie configurations. |
