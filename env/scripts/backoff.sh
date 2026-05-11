#!/bin/bash
# Retries a command a with backoff.
# Successive backoffs double the timeout.
# Beware of set -e killing your whole script!
# usage:  with_backoff {"command"}
function with_backoff {
  local max_attempts=5
  local timeout=2
  local attempt=0
  local exitCode=0

  while [[ $attempt < $max_attempts ]]
  do
    "$@"
    exitCode=$?

    if [[ $exitCode == 0 ]]
    then
      break
    fi

    echo "Failure! Retrying in $timeout.." 1>&2
    sleep $timeout
    attempt=$(( attempt + 1 ))
    timeout=$(( timeout + $BOLT_TASK_ROLE_RANK ))
    timeout=$(( timeout * 2 ))
  done

  if [[ $exitCode != 0 ]]
  then
    echo "You've failed me for the last time! ($@)" 1>&2
  fi

  return $exitCode
}