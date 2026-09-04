# Git and Production Workflow

This repository uses two long-lived branches:

| Branch | Purpose | Railway behavior |
| --- | --- | --- |
| `develop` | Normal development, testing, and review | Never deployed to production |
| `main` | Reviewed production releases only | Railway's production service tracks this branch |

Railway auto-deploy is disabled. Pushing or merging code does not replace the running production
version. A reviewed `main` revision must be deployed manually in Railway.

## Start a development session

Confirm that the local checkout is on `develop`, then synchronize it:

```bash
git switch develop
git pull --ff-only origin develop
git status
```

The status output should begin with `## develop...origin/develop`. Do not begin normal work while
the checkout is on `main`.

For a small change, commit directly to `develop`:

```bash
# Edit files and run the relevant checks.
git status
git add <files-you-intend-to-commit>
git commit -m "Describe the change"
git push origin develop
```

Avoid `git add .` when unrelated files or generated previews are present. Review `git status` and
stage only the intended files.

For a larger or experimental change, create a short-lived feature branch from `develop`:

```bash
git switch develop
git pull --ff-only origin develop
git switch -c feature/<short-description>

# Edit, test, commit, and push.
git push -u origin feature/<short-description>
```

Open a pull request from the feature branch into `develop`. Delete the feature branch after it is
merged.

## Test before a production release

Run checks from a clean, up-to-date `develop` checkout:

```bash
pytest
newsletter-agent validate-config
newsletter-agent sample-preview --output outputs
newsletter-agent run --dry-run
```

`run --dry-run` may read Gmail and call configured model providers, but it does not create Gmail
drafts, remove Gmail labels, or write to PostgreSQL. If a real draft is needed for inspection while
preserving labels, use `newsletter-agent run --keep-labels` deliberately.

Before release, also review:

- `git status` is clean;
- configuration contains the intended enabled profiles and recipients;
- `.env` is not tracked;
- tests and configuration validation pass;
- sample or dry-run newsletters have the expected content and layout;
- any database schema change includes an Alembic migration.

## Release `develop` to production

1. Push the final development commits:

   ```bash
   git switch develop
   git pull --ff-only origin develop
   git push origin develop
   ```

2. On GitHub, open a pull request with:

   - base branch: `main`;
   - compare branch: `develop`.

3. Review the file diff and confirm the test results, configuration changes, migration changes,
   and absence of secrets.
4. Merge the pull request into `main`.
5. In Railway, open the `newsletter-agent` production service and deliberately deploy the latest
   commit from `main`.
6. Confirm that the deployment succeeds, Alembic completes, the production schedule remains
   Thursday at `14:30 UTC` (`20:00 IST`), and the production activation variable remains `true`.

Because Railway auto-deploy is disabled, merging the pull request does not deploy by itself.

### Deploy is not the same as Run now

- **Deploy latest commit** updates the production image to the reviewed `main` revision.
- **Run now** immediately executes the active cron service. With production enabled, that run can
  create Gmail drafts and remove successfully processed Gmail labels.

Do not select **Run now** merely to deploy code. Use it only when an immediate production
newsletter run is intentionally required.

## After a release

Synchronize `develop` with the released `main` branch so both long-lived branches share the same
release history:

```bash
git switch develop
git pull --ff-only origin develop
git merge origin/main
git push origin develop
```

If Git reports a conflict, stop and resolve it carefully before pushing. Do not force-push either
long-lived branch.

## If work starts on `main` accidentally

If no commit has been created, move the working changes to `develop`:

```bash
git status
git switch develop
```

Git normally carries uncommitted changes across when there is no conflict. Confirm the files with
`git status` before continuing.

If commits were created on `main` but have not been pushed, do not reset or delete them. Preserve
them on a new branch and ask for a review before changing branch history:

```bash
git switch -c recovery/<short-description>
git push -u origin recovery/<short-description>
```

If a commit is accidentally pushed to `main`, Railway will not auto-deploy it. Review the commit
and either accept it as a release or revert it through a new commit. Do not force-push `main`.

## Roll back production

If a newly deployed version fails:

1. Do not use **Run now** again.
2. Open the Railway deployment history for `newsletter-agent`.
3. Select the last known-good successful deployment and use Railway's redeploy/rollback action.
4. Confirm the previous deployment becomes active and review its logs.
5. Fix the problem on `develop`, test it, and follow the normal release process.

A database migration may require a separate forward-fix migration. Do not manually reverse or
delete production database records without a reviewed recovery plan.

## Current enforcement limitation

GitHub reports that rulesets are not enforceable for this private repository on the current
account tier. Direct pushes to `main` therefore cannot currently be blocked by GitHub. The active
safeguards are:

- development occurs on `develop` or a feature branch;
- `main` is used only for reviewed releases;
- Railway tracks only `main`;
- Railway auto-deploy is disabled;
- production deployment is a deliberate manual action.

If the repository moves to a GitHub plan that supports rulesets for private repositories, add a
`main` ruleset that requires pull requests, blocks force pushes and deletion, and requires the test
workflow to pass.

