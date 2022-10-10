"""
Download event from github public repo.

# Improvement :
- get start_date and end_date for each event (by extension we can have duration)
- get commit maker
- get commit count
- get comments count
"""

import logging
import os
import datetime
from tracemalloc import start
import tqdm
import json
import argparse
import pandas as pd
import time
from enum import Enum
import sys
import traceback
import sqlite3

from github import Github

class TqdmStream(object):
    """
    Streamer to properly display logging message during a tqdm display
    """
    @classmethod
    def write(_, msg):
        tqdm.tqdm.write(msg, end='')

    @classmethod
    def flush(_):
        sys.stdout.flush()

def set_logger(file_name=None, log_level=logging.DEBUG):
    # create logger for prd_ci
    """
    Args:
        file_name: name for the file to log (full path and extension ex: output/run.log)
        :param log_level:
    """
    log = logging.getLogger('main')
    log.propagate = False
    log.setLevel(level=logging.DEBUG)

    # create formatter and add it to the handlers
    formatter = logging.Formatter('[%(levelname)s] (%(asctime)s) at %(lineno)dl : %(message)s')

    if file_name:
        # create file handler for logger.
        fh = logging.FileHandler(file_name, mode='w+')
        fh.setLevel(level=logging.DEBUG)
        fh.setFormatter(formatter)
    # create console handler for logger.
    ch = logging.StreamHandler(TqdmStream)
    ch.setLevel(level=logging.ERROR)
    ch.setFormatter(formatter)

    # add handlers to logger.
    if file_name:
        log.addHandler(fh)

    log.addHandler(ch)
    log.setLevel(level=logging.DEBUG)


    return log

class EventType(Enum):
    """
    Define event type in github repo
    TIPS : ONLYPR doesn't seems to exist. Maybe by design a PR is linked to an issue (the comment part)
    """
    ONLYISSUE = 0
    ONLYPR = 1
    ISSUEPR = 2

class Event:
    """
    Event object, as a pull request or an issue. It will hold all participants to this event (owner, assignes, comments)
    TODO: What about commits ?
    """
    repo: str  # repo name
    id: int  # unique id
    nbr: int  # PR/issue number to link to the website
    etype: EventType  # type of event (pr or issue)
    participants: set  # set of participant

    def __init__(self, repo, id, nbr, etype=0) -> None:
        self.repo = repo
        self.id = id
        self.nbr = nbr
        self.etype = etype
        self.participants = set()

    def to_dataframe(self):
        return pd.DataFrame({'repo': self.repo, 'id': self.id, 'nbr': self.nbr, 'event_type': self.etype, 'participants': list(self.participants)})


    def to_list(self):
        return [self.repo, self.id, self.nbr, self.etype, self.participants]

def init_db(name='OSCP_data.db'):
    con = sqlite3.connect(name)
    #cur = con.cursor()
    #cur.execute(f"DROP TABLE IF EXISTS event ;")
    return con


def check_remaining_request(g):
    """
    Github API only authorize 60 request for non authenticated user and 5000 for one.
    This method check the remaining request and if the limit is reached. The process will sleep until it can
    make new request
    """
    log = logging.getLogger('main')

    remaining = g.rate_limiting[0]
    # TODO improvement: encapsulate Github class with this method to check each time a request is made and sleep if it's needed
    if remaining < 4:  # a little margin is taken because we are not sure to check each time a request if done
        log.critical(f'Time to sleep or change internet {remaining} requests')
        end_sleep = datetime.datetime.fromtimestamp(g.rate_limiting_resettime)
        start_sleep = datetime.datetime.now()
        log.critical(f'Sleep from {start_sleep.strftime("%Y-%m-%dT%Hh%Mm%Ss")} to {end_sleep.strftime("%Y-%m-%dT%Hh%Mm%Ss")}')
        duration = int((end_sleep - start_sleep).total_seconds()) + 10
        if duration < 0 :
            logging.critical(f"Problem with duration {duration}")
            time.sleep(360)
        else:
            time.sleep(duration)

        log.critical('Waiting done, go again')
        check_remaining_request(g)

def get_from_named_user(named_user):
    """
    Extract user login and id from User class
    """
    return (named_user.id, named_user.login)

def get_event_from_pr(pr, repo, g):
    """
    Explore a PullRequest to find all contributor
    # TODO what about commits ?
    """
    check_remaining_request(g)

    # Find event type
    if pr.as_issue():
        typ = EventType.ISSUEPR.value
    else:
        typ = EventType.ONLYPR.value
    ev  = Event(repo=repo, id=pr.id, nbr=pr.number, etype=typ)

    check_remaining_request(g)
    ev.participants.add(get_from_named_user(pr.user))

    assignes = tuple(get_from_named_user(user) for user in pr.assignees)
    if assignes:
        if type(assignes) == list:
            for a in assignes:
                ev.participants.add(a)
        else:
            ev.participants.add(assignes)

    check_remaining_request(g)
    # comments
    if pr.comments:
        com = pr.get_comments()
        for c in com:
            ev.participants.add(get_from_named_user(c.user))

    check_remaining_request(g)
    if pr.review_comments:
        com = pr.get_review_comments()
        for c in com:
            ev.participants.add(get_from_named_user(c.user))

    check_remaining_request(g)
    com = pr.get_issue_comments()
    if com.totalCount:
        for c in com:
            ev.participants.add(get_from_named_user(c.user))

    # commits
    check_remaining_request(g)
    if pr.commits:
        commits = pr.get_commits()
        for c in commits:
            ev.participants.add(get_from_named_user(c.author))

    return ev

def get_repo_todo(conn):
    query = "SELECT"


def checkpoint(df, repo, conn, ids, moment):
    """
    Checkpoint to save data in the DB because of raw overflow
    """
    log = logging.getLogger('main')

    log.warning(f'Checkpoint save at {moment} for {repo}')
    df.participants = df.participants.astype('string')
    count = df.to_sql(name='event', con=conn, index=False, if_exists = 'append', dtype='string')
    log.debug(f"{count} rows added to event")
    ids.union(set(df.id.to_list()))
    df = df.iloc[0:0]
    return df, ids

def error_log(log, err, sys_stack, repo_missing_path, repo, type):
    """
    Log error with stack trace
    """
    trace = sys_stack[2].tb_lineno
    log.error(err, sys_stack[1], trace)
    with open(repo_missing_path, 'a+') as fd:
        fd.write(f"{repo}, {type}\n")

def main(cpr=None):
    now = datetime.datetime.now().strftime("%Y-%m-%dT%Hh%Mm%Ss")

    # Argument management
    parser = argparse.ArgumentParser(description='github scrapping')
    parser.add_argument('-o', '--output', required=True,
                        help='for output log')
    parser.add_argument('-t', '--token', required=False,
                        help='token for github')
    parser.add_argument('-r', '--run_id', required=False,
                        help='run_id for db')
    parser.add_argument('-db', '--database', required=False, default='OSCP_data.db',
                        help='database for all data')

    args = parser.parse_args(cpr)
    root = args.output

    #  Logging managment
    log_file = os.path.join(root, fr'github_dl_{now}.log')
    set_logger(log_file)
    log = logging.getLogger('main')
    log.critical("start")

    #  connect to db
    conn = init_db(args.database)

    # Get list of repo
    cursor = conn.cursor()
    query = f"SELECT name FROM repo WHERE token_id = {args.run_id} AND done = {0}"
    repos = [ret[0] for ret in cursor.execute(query).fetchall()]
    cursor.close()

    #  Define path for input/output file
    repo_missing_path =  os.path.join(root, r'repos_name_missing.txt')

    #  Load github repo name to reach for data
    log.debug(f"{len(repos)} repos to do")

    ### Main process
    g = Github(login_or_token=args.token)  # init github conenction
    for repo in tqdm.tqdm(repos,desc="Repos", leave=True, position=0):  # iterate over all repos
        df = pd.DataFrame(columns=['repo', 'id', 'nbr', 'event_type', 'participants'])
        ids = set()
        #  Connect to repo
        try:
            log.info(f'Doing repo {repo}')
            check_remaining_request(g)
            rep = g.get_repo(repo)

        except Exception as err:
            error_log(log, err, sys.exc_info(), repo_missing_path, repo, 'get_repo')
            continue

        # Get all PR from the repo
        try:
            log.debug('Take pull request')
            check_remaining_request(g)
            prs = rep.get_pulls(state='all')  # get all pr
            cpt = 0
            for pr in tqdm.tqdm(prs, total=prs.totalCount, desc="PR", leave=False, position=1):
                check_remaining_request(g)
                ev = get_event_from_pr(pr, repo, g)
                df = pd.concat([df, ev.to_dataframe()], ignore_index=True)

                #  checkpoint to save in sqlite
                if cpt > 50:
                    df, ids = checkpoint(df, repo, conn, ids, 'pr')
                    cpt = 0
                cpt += 1

            ids.union(set(df.id.to_list()))
            df.participants = df.participants.astype('string')
            df.to_sql(name='event', con=conn, index=False, if_exists = 'append', dtype='string')
            ids.union(set(df.id.to_list()))
            df = df.iloc[0:0]

        except Exception as err:
            error_log(log, err, sys.exc_info(), repo_missing_path, repo, 'pr')

        # Get all issues from the repo
        try:
            log.debug('Take issues')
            check_remaining_request(g)
            issues = rep.get_issues(state='all')

            cpt = 0
            for iss in tqdm.tqdm(issues,desc="Issue", total=prs.totalCount, leave=False, position=1):

                pr = iss.pull_request
                ev = Event(repo=repo, id=iss.id, nbr=iss.number)

                if pr:
                    ev.etype = EventType.ISSUEPR.value
                    check_remaining_request(g)
                    pr = iss.as_pull_request()
                    if pr.id in ids:
                        continue
                    else:
                        check_remaining_request(g)
                        evpr = get_event_from_pr(pr, repo, g)
                        ev.participants = evpr.participants
                    df = pd.concat([df, ev.to_dataframe()], ignore_index=True)

                else:
                    ev.etype = EventType.ONLYISSUE.value
                    check_remaining_request(g)
                    ev.participants.add(get_from_named_user(iss.user))
                    check_remaining_request(g)
                    assignes = tuple(get_from_named_user(user) for user in iss.assignees)
                    if assignes:
                        ev.participants.add(assignes)

                    # comments
                    check_remaining_request(g)
                    com = iss.get_comments()
                    for c in com:
                        check_remaining_request(g)
                        ev.participants.add(get_from_named_user(c.user))
                    df = pd.concat([df, ev.to_dataframe()], ignore_index=True)

                if cpt > 50:
                    df, ids = checkpoint(df, repo, conn, ids, 'issue')
                    cpt = 0
                cpt += 1
            
            log.debug(f"Saving {repo}")
            df, ids = checkpoint(df, repo, conn, ids, 'end')

            query = f"UPDATE repo SET done = {1} WHERE name = \"{repo}\""
            log.debug(f"Query update : {query}")
            cursor = conn.cursor()
            cursor.execute(query)
            conn.commit()
            cursor.close()
            log.debug(f"{repo} finished")

        except Exception as err:
            error_log(log, err, sys.exc_info(), repo_missing_path, repo, 'issue')

    log.info("end")

       
if __name__ == "__main__":
    #args = "-o .\output -r 0 -t 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"
    main() #args.split(' '))