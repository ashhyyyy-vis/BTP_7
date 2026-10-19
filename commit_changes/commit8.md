# Commit Changes

## **Title:** Commit 8 : Add WSL2 config guidelines, user notification directives, and Google Drive / rclone bundle workflow to runbook

### **Summary of Changes:**
1. **WSL2 Configuration & Agent Notification Directives**:
   - Added explicit instructions for configuring `.wslconfig` (`memory=16GB`, `processors=8`, `swap=8GB`) on the Windows host to prevent memory throttling during Whisper large-v3 evaluation.
   - Instructed the AI agent to verify WSL2 and notify the user with clear setup steps if `.wslconfig` or GPU pass-through is unconfigured.
2. **Google Drive & rclone Bundle Transfer**:
   - Updated Step 2 with Google Drive download workflows (`gdown`, `rclone`, and Windows host Downloads copy).
   - Streamlined checksum validation by generating `baseline_bundle.md5` directly via `echo` on the target machine without needing a separate checksum download.

