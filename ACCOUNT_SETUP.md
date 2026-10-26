# Accounts and watchlists

Render now targets Firebase. See `FIREBASE_SETUP.md` for the current deployment
instructions. The MySQL section below describes the optional legacy backend.

## Local development

Build the React frontend with `npm run build` in `frontend`, then run
`python -m uvicorn app:app --host 127.0.0.1 --port 8000` from the project root.
Open http://127.0.0.1:8000. No account service configuration is needed locally.

Local accounts, sessions, and watchlists persist in `data/accounts/accounts.sqlite3`.
This file must never be committed or copied into a deployment image.
The existing anonymous recent-history list remains browser-local; it is not a watchlist.

Local users register with a username (3-32 ASCII letters, digits, underscores) and
a nonempty password without app-specific length limits. Firebase accounts remain
subject to the project's Firebase password policy. Usernames are case insensitive. Sessions last seven
days and logout revokes the server-side session. Password reset/email recovery
and account deletion are not implemented in this version.

## Render

Render's free web filesystem is ephemeral. Accounts therefore require an external
MySQL database. No paid database is automatically provisioned by this change.
Until configured, the dashboard still works, but account actions are unavailable.
Stock data can continue using CSV with `MYSQL_ENABLED=0`.

Set these values in Render's Environment page, never in Git:

The Blueprint declares connection values with `sync: false`. Render only prompts
for these when creating a Blueprint. For an existing service, add them manually
under Environment, then save and deploy. Updating YAML alone does not connect it.
Use a database hostname reachable from Render, not `localhost` or `127.0.0.1`:
those addresses refer to the Render container, not your Windows PC.

```text
AUTH_PRODUCTION=1
AUTH_DB_BACKEND=mysql
AUTH_MYSQL_HOST=<external database hostname>
AUTH_MYSQL_PORT=3306
AUTH_MYSQL_DATABASE=<existing database name>
AUTH_MYSQL_USER=<dedicated application user>
AUTH_MYSQL_PASSWORD=<strong database password>
```

The database must already exist. The application creates four `account_*` tables
on first use. Give the application user CREATE, SELECT, INSERT, UPDATE, DELETE
permissions on this database; do not use a root account. Production connections
verify TLS certificates and hostnames. If the provider uses a private CA, set
`AUTH_MYSQL_SSL_CA` to its CA certificate file path (Render Secret File).

Same-origin frontend/API is the default. `AUTH_ALLOWED_ORIGINS` can contain a
comma-separated list of additional exact origins, including scheme and port.
Never use `*`. Production cookies require HTTPS.

After deployment verify signup, logout/login, saving/removing a stock and
refreshing the page. Restart the service and verify the saved list persists.
Use two accounts to confirm their watchlists are separate. Back up the external
database independently of CSV/model backups.

## Tests

```text
python -m unittest discover -s tests -p test_accounts.py -v
```

The tests use temporary SQLite storage and do not create real user accounts.
External MySQL connectivity must also be verified in the target environment.
