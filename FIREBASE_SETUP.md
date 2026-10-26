# Firebase on Render

Project: `stockproject-736df`. Public web configuration is in
`database/firebase_client.py`. Analytics and Cloud Storage are not used.

## Console setup (required before deployment)

1. Keep the Firebase Spark plan. Enable Authentication > Email/Password.
2. Create the default Firestore Standard database in production mode.
3. Publish `firestore.rules`. It denies ALL browser reads/writes. FastAPI uses
   Admin SDK with IAM permissions and checks session ownership on every request.
4. Configure a dedicated service account for the Render app with Firebase
   Authentication Admin and Cloud Datastore User permissions. Do not grant Owner.
5. Generate its JSON key and upload it to Render > Environment > Secret Files,
   named `firebase-service-account.json`. Never send the key through chat, Git,
   frontend environment variables, or screenshots. Rotate exposed keys.
6. Set these Render environment variables (existing services need manual updates):

```text
AUTH_PRODUCTION=1
AUTH_DB_BACKEND=firebase
STOCK_DB_BACKEND=firebase
GOOGLE_APPLICATION_CREDENTIALS=/etc/secrets/firebase-service-account.json
MYSQL_ENABLED=0
```

Remove unused MYSQL_PASSWORD / AUTH_MYSQL_PASSWORD from Render. Deploy the updated
code after setting secrets. The public web configuration alone is not enough.
Use the same-origin frontend and API over HTTPS. The project ID in the secret
must match the configured project. No test-mode/public Firestore rules are needed.

## Local development

Put the service account key in ignored `.secrets/firebase-service-account.json`.
In PowerShell set:

```powershell
$env:GOOGLE_APPLICATION_CREDENTIALS="$PWD\.secrets\firebase-service-account.json"
$env:AUTH_DB_BACKEND="firebase"
$env:STOCK_DB_BACKEND="firebase"
$env:AUTH_PRODUCTION="0"
python -m uvicorn app:app --host 127.0.0.1 --port 8011
```

With no AUTH_DB_BACKEND setting, the existing local SQLite accounts still work.
Missing Firebase credentials disable account actions, not the stock dashboard.

## Accounts and data migration

Firebase login uses a real email address instead of the old username. Existing
SQLite/MySQL users and their password hashes are NOT automatically migrated.
Keep the old database and establish a verified old-user to Firebase UID mapping
before copying watchlists. Do not infer ownership from matching display names.
New accounts store no passwords in Firestore. Firebase Auth manages passwords;
HttpOnly session cookies last 7 days. Each cookie has a hashed allowlist entry in
Firestore, so logout invalidates that session immediately. Logout does not log
out other devices. ID tokens are held only in memory during the login exchange.
An account can save up to 100 stocks. Existing browser history stays browser-local.

The stock dashboard uses a single Firestore document snapshot, cached for 60
seconds per process. CSV remains the fallback, and fresher CSV predictions win.
This is NOT a complete migration of historical MySQL tables or training data.
Historical prices, features, labels, model files and old logs remain unchanged.

Preview the dashboard import without contacting Firebase:

```powershell
python -m database.publish_firestore
```

After checking the date/count/size, publish from a trusted machine with the secret:

```powershell
python -m database.publish_firestore --apply
```

Run again after completing a price update and prediction run. Publishing is not
yet wired to the existing scheduler. It atomically updates the snapshot and
rejects an older date. It does not erase historical data. The initial dry run
contained 300 stocks and 299 predictions, about 94 KB, using one document write.

Firestore Standard free quotas include 1 GiB storage, 50,000 document reads/day,
20,000 writes/day. Authentication also has provider quotas. Do not enable billing
or paid TTL/backup features for this setup. Expired session documents must be
periodically cleaned up by a trusted script; expiration is enforced even if a
document has not been deleted. Monitor usage before importing historical data.

## Verification

```text
python -m unittest discover -s tests -p test_*accounts.py -v
```

These tests use SQLite and mocked Firebase boundaries, not the live project.
After configuring the real project, verify signup, login/logout, two-user
isolation, watchlist persistence after restart, expired sessions, and CSV fallback.
`/api/auth/me` exposes only public web config and readiness, never the private key.
Test-mode rules must never be deployed. No cloud deployment or user migration is
complete until verified against the configured Firebase project and Render URL.
