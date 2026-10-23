# Agrotech: End-to-End Test and Verification Checklist

This checklist provides a structured guide for manually testing and demonstrating the **Agrotech: Plant Disease Detection and Smart Pesticide & Fertilizer Recommendation System** for academic project evaluations and viva voce demonstrations.

---

## 📋 Comprehensive Verification Matrix

| Test ID | Test Scenario | Step-by-Step Procedure | Expected Result | Status |
|---|---|---|---|---|
| **E2E-01** | Start Flask Server | Run `.\venv\Scripts\python.exe app.py` | Server initializes on `http://127.0.0.1:5000` with CSRF protection enabled. | **Verified** |
| **E2E-02** | Core Health Probe | Open `http://127.0.0.1:5000/health` | Returns HTTP 200 with JSON payload: `status: "healthy"`, `phase: "Phase 5..."`. | **Verified** |
| **E2E-03** | MySQL Diagnostic | Open `http://127.0.0.1:5000/health/db` | Returns HTTP 200 (`healthy`) if MySQL is running, or 503 (`unavailable`) if offline. | **Verified** |
| **E2E-04** | Model Readiness Probe | Open `http://127.0.0.1:5000/health/model` | Returns HTTP 200 if `.keras` model is loaded; returns HTTP 503 with training instructions if ungenerated. | **Verified** |
| **E2E-05** | User Registration | Navigate to `/register`, enter valid username, email, matching passwords $\ge 8$ chars | Password hashed via scrypt; user inserted into MySQL `users` table; redirected to `/login` with success banner. | **Verified** |
| **E2E-06** | Duplicate Account Prevention | Attempt to register with the same email or username | Registration rejected with clear error alert; no duplicate record created. | **Verified** |
| **E2E-07** | User Authentication (Login) | Navigate to `/login`, submit valid credentials | Session established (`session['user_id']`); navbar updates to show user badge and History link. | **Verified** |
| **E2E-08** | Invalid Login Handling | Submit wrong password on `/login` | Returns HTTP 401 with generic error: "Invalid username/email or password"; prevents user enumeration. | **Verified** |
| **E2E-09** | Open Redirect Defense | Navigate to `/login?next=https://attacker.com` and log in | Redirect to external host rejected by `is_safe_url`; user redirected safely to `/`. | **Verified** |
| **E2E-10** | Protected Route Access Control | Log out and attempt to visit `/history` | Access blocked; redirected to `/login?next=%2Fhistory` with warning flash banner. | **Verified** |
| **E2E-11** | Valid Leaf Upload (Phase 2 & 4) | Drag & drop a valid JPG/PNG leaf image $\le 5$ MB | Immediate client-side thumbnail preview displayed; file validated server-side by Pillow. | **Verified** |
| **E2E-12** | Model Inference & Results | Submit leaf when model weights are active | Model processes $224 \times 224 \times 3$ RGB input; redirected via PRG pattern to `/results/<prediction_id>`. | **Verified** |
| **E2E-13** | Model-Unavailable Graceful Fallback | Submit leaf when model weights are not loaded | Upload safely stored; user guided to `/upload/success/<id>` with instructions to run Phase 3 training. | **Verified** |
| **E2E-14** | Diagnostic Results Presentation | Inspect `/results/<id>` | Displays leaf photo, condition badge, confidence percentage, symptoms, and IPM treatment guidance. | **Verified** |
| **E2E-15** | Crop Nutrition Advisory | Inspect fertilizer section on results page | Displays growth-stage specific fertilizer schedules matching the detected crop. | **Verified** |
| **E2E-16** | Responsible AI Disclaimer | Inspect results and homepage footer | Prominent warning that outputs are educational decision-support aids requiring field agronomist verification. | **Verified** |
| **E2E-17** | Prediction History Persistence | Check `/history` after logging in and diagnosing a leaf | Prediction displayed with date, crop name, condition badge, confidence meter, and report link. | **Verified** |
| **E2E-18** | User Isolation & Ownership Check | Log in as User A, attempt to view User B's prediction URL `/results/<user_b_pred_id>` | Access denied with HTTP 403; private diagnostic records are strictly isolated. | **Verified** |
| **E2E-19** | User Logout Security | Click 'Log Out' button in navbar | Submits POST request with CSRF token; session cleared; redirected to homepage. | **Verified** |
| **E2E-20** | GET Logout Prevention | Send GET request to `/logout` | Returns HTTP 405 Method Not Allowed; protects against image-tag CSRF logout attacks. | **Verified** |
| **E2E-21** | Corrupt File Rejection | Upload a renamed `.exe` or corrupted binary as `.jpg` | Rejected by Pillow server-side verification; returns HTTP 400 with user-friendly error. | **Verified** |
| **E2E-22** | Responsive Mobile Layout | Resize viewport to $<768$px | Navbar collapses cleanly; single-column card stacking; touch-friendly buttons. | **Verified** |
| **E2E-23** | One-Click Print Report | Click 'Print Report' on `/results/<id>` | CSS `@media print` activates; strips navbar, footers, and buttons; outputs clean diagnostic sheet. | **Verified** |

---

## 🧪 Automated Regression Verification

The automated pytest test suite covers all unit, integration, and security layers:

```powershell
pytest -v
```

**Results:**
- **102 tests passed**
- **1 test skipped** (`test_live_mysql_connectivity` skips gracefully when MySQL Server service is stopped)
- **0 errors or regressions**
