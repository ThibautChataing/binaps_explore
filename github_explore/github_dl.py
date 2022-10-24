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
from xml.dom.xmlbuilder import DOMEntityResolver
import tqdm
import json
import argparse
import pandas as pd
import time
from enum import Enum
import sys
from collections.abc import Iterable
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

class ContribType(Enum):
    dev = 0
    comment = 1


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
    contrib: ContribType

    def __init__(self, repo, id, nbr, etype=0) -> None:
        self.repo = repo
        self.id = id
        self.nbr = nbr
        self.etype = etype
        self.participants = set()

    def clean_for_savepoint(self):
        self.participants = set()



    def to_dataframe(self):
        p_id = []
        p_login = []
        p_name = []
        p_mail = []
        p_contrib = []
        for p in self.participants:
            p_id.append(p[0])
            p_login.append(p[1])
            p_name.append(p[2])
            p_mail.append(p[3])
            p_contrib.append(p[4])
        df = pd.DataFrame({'repo': self.repo, 'id': self.id, 'nbr': self.nbr, 'event_type': self.etype, 'p_id': list(p_id), 'p_login': list(p_login), 'p_name': list(p_name), 'p_mail': list(p_mail), 'contrib_type': p_contrib})
        df = df.infer_objects()
        return df


    def add_participants(self, participants, contrib_type):
        if isinstance(participants, Iterable):
            if isinstance(participants[0], Iterable):
                for p in participants:
                    self.participants.add((*p, contrib_type.value))
            else:
                self.participants.add((*participants, contrib_type.value))


def init_db(name='OSCP_data.db'):
    con = sqlite3.connect(name)
    # cur = con.cursor()
    # cur.execute(f"DROP TABLE IF EXISTS event ;")
    # con.commit()
    # cur.close()
    return con


class CheckpointManager:
    def __init__(self, conn) -> None:
        self.conn = conn
        self.event_ids = set()

    def health_check(self, g, ev=None, moment='', force_save=False):
        
        log = logging.getLogger('main')
        log.debug(f"Health check at {moment}")
        if isinstance(ev, Event):
            df = ev.to_dataframe()
            if (len(df) > 100) or force_save:
                df = self.date_checkpoint(df, moment)
                ev.clean_for_savepoint()

        self.check_remaining_request(g)
        return ev


    def date_checkpoint(self, df, moment):
        """
        Checkpoint to save data in the DB because of raw overflow
        """
        log = logging.getLogger('main')

        log.debug(f'Data checkpoint at {moment} for {df.repo.unique()}')
        count = df.to_sql(name='event', con=self.conn, index=False, if_exists = 'append', dtype='string')
        log.debug(f"{count} rows added to event")
        self.event_ids.union(set(df.id.to_list()))
        df = df.iloc[0:0]
        return df

    def check_remaining_request(self, g):
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
            g.get_rate_limit()
            self.check_remaining_request(g)


def get_from_named_user(named_user):
    """
    Extract user login and id from User class
    """
    key = ['id', 'login', 'name', 'email']
    ret = []
    for k in key:
        e = getattr(named_user, k, -1)
        if e is None:
            e = -1
        ret.append(e)
    return tuple(ret)

def get_event_from_pr(pr, repo, g, health_check, based_ev=None):
    """
    Explore a PullRequest to find all contributor
    # TODO what about commits ?
    """
    health_check.health_check(g=g, moment='Get event from pr')
    log = logging.getLogger('main')
    if not based_ev:
        # Find event type
        if pr.as_issue():
            typ = EventType.ISSUEPR.value
        else:
            typ = EventType.ONLYPR.value
        ev  = Event(repo=repo, id=pr.id, nbr=pr.number, etype=typ)
    else:
        ev = based_ev

    

    health_check.health_check(g=g, moment='Get event from pr')

    ev.add_participants(get_from_named_user(pr.user), contrib_type=ContribType.dev)

    assignes = tuple(get_from_named_user(user) for user in pr.assignees)
    if assignes:
        ev.add_participants(assignes, contrib_type=ContribType.dev)

    ev = health_check.health_check(ev=ev, g=g, moment='Get event from pr')
    # comments
    if pr.comments:
        com = pr.get_comments()
        for c in com:
            if c:
                ev.add_participants(get_from_named_user(c.user), contrib_type=ContribType.comment)
            else:
                logging.warning(f"contributor missing in comments")

    ev = health_check.health_check(ev=ev, g=g, moment='Get event from pr')
    if pr.review_comments:
        com = pr.get_review_comments()
        for c in com:
            if c:
                ev.add_participants(get_from_named_user(c.user), contrib_type=ContribType.comment)
            else:
                logging.warning(f"contributor missing in review_comment")

    ev = health_check.health_check(ev=ev, g=g, moment='Get event from pr')
    com = pr.get_issue_comments()
    if com.totalCount:
        for c in com:
            if c:
                ev.add_participants(get_from_named_user(c.user), contrib_type=ContribType.comment)
            else:
                logging.warning(f"contributor missing in issue comment")

    # commits
    ev = health_check.health_check(ev=ev, g=g, moment='Get event from pr')
    if pr.commits:
        commits = pr.get_commits()
        for c in commits:
            if c.author:
                ev.add_participants(get_from_named_user(c.author), contrib_type=ContribType.dev)
            elif c.commit.author:
                ev.add_participants(get_from_named_user(c.commit.author), contrib_type=ContribType.dev)
            else:
                logging.warning(f"contributor missing in commit")

    ev = health_check.health_check(ev=ev, g=g, moment='Get event from pr', force_save=True)
    return health_check


def error_log(log, err, sys_stack, repo_missing_path, repo, type):
    """
    Log error with stack trace
    """
    log = logging.getLogger('main')
    log.critical(f"{err} - {sys_stack[2].tb_lineno}")
    with open(repo_missing_path, 'a+') as fd:
        fd.write(f"{repo}, {type}\n")

def get_todo_repos(conn, run_id):
    limit = 1
    log = logging.getLogger("main")
    # Get list of repo
    cursor = conn.cursor()
    log.debug("choose repos")
    q = f"UPDATE repo SET token_id={run_id} WHERE id IN (SELECT id FROM repo WHERE token_id=-1 LIMIT {limit})"
    cursor.execute(q)
    conn.commit()

    query = f"SELECT name FROM repo WHERE token_id = {run_id} AND done = {0} LIMIT {limit}"
    repos = [ret[0] for ret in cursor.execute(query).fetchall()]
    cursor.close()
    log.debug(f"find {len(repos)} to do")

    return repos


def get_data(repos, repo_missing_path, g, conn, health_check):
    log = logging.getLogger("main")
    for repo in tqdm.tqdm(repos,desc="Repos", leave=True, position=0):  # iterate over all repos
        health_check.event_ids = set()

        #  Connect to repo
        try:
            log.info(f'Doing repo {repo}')
            health_check.health_check(g=g, moment='start_repo')
            rep = g.get_repo(repo)

        except Exception as err:
            error_log(log, err, sys.exc_info(), repo_missing_path, repo, 'get_repo')
            continue

        # Get all PR from the repo
        try:
            log.debug('Take pull request')
            health_check.health_check(g=g, moment='start PR')
            prs = rep.get_pulls(state='all')  # get all pr
            log.debug(f"{prs.totalCount} prs found")
            for pr in tqdm.tqdm(prs, total=prs.totalCount, desc="PR", leave=False, position=1):
                health_check = get_event_from_pr(pr=pr, repo=repo, g=g, health_check=health_check)
                break


        except Exception as err:
            log.critical('PR')
            error_log(log, err, sys.exc_info(), repo_missing_path, repo, 'pr')

        # Get all issues from the repo
        try:
            log.debug('Take issues')
            health_check.health_check(g=g, moment="Issue")
            issues = rep.get_issues(state='all')

            log.debug(f"{issues.totalCount} issues found")
            for iss in tqdm.tqdm(issues,desc="Issue", total=issues.totalCount, leave=False, position=1):

                pr = iss.pull_request
                ev = Event(repo=repo, id=iss.id, nbr=iss.number)

                if pr:
                    ev.etype = EventType.ISSUEPR.value
                    health_check.health_check(g=g, moment='pr in issue')
                    pr = iss.as_pull_request()
                    if pr.id in health_check.event_ids:
                        continue
                    else:
                        health_check = get_event_from_pr(pr, repo, g, health_check=health_check, based_ev=ev)

                else:
                    ev.etype = EventType.ONLYISSUE.value

                    health_check.health_check(g)
                    ev.add_participants(get_from_named_user(iss.user), contrib_type=ContribType.comment)

                    ev = health_check.health_check(g=g, ev=ev, moment='issue user')
                    assignes = tuple(get_from_named_user(user) for user in iss.assignees)
                    if assignes:
                        ev.add_participants(assignes, contrib_type=ContribType.comment)

                    # comments
                    ev = health_check.health_check(g=g, ev=ev, moment='issue comment')
                    com = iss.get_comments()
                    for c in com:
                        ev.add_participants(get_from_named_user(c.user), contrib_type=ContribType.comment)
                        ev = health_check.health_check(g=g, ev=ev, moment='issue comment')
                ev = health_check.health_check(g=g, ev=ev, moment='issue comment', force_save=True)
                break
                

        except Exception as err:
            log.critical('ISSUE')
            error_log(log, err, sys.exc_info(), repo_missing_path, repo, 'issue')

        try:    
            log.debug(f"Saving {repo}")

            query = f"UPDATE repo SET done = {1} WHERE name = \"{repo}\""
            log.debug(f"Query update : {query}")
            cursor = conn.cursor()
            cursor.execute(query)
            conn.commit()
            cursor.close()
            log.debug(f"{repo} finished")
        
        except Exception as err:
            log.critical('END')
            error_log(log, err, sys.exc_info(), repo_missing_path, repo, 'end')

def main(cpr=None):
    now = datetime.datetime.now().strftime("%Y-%m-%dT%Hh%Mm%Ss")

    # Argument management
    parser = argparse.ArgumentParser(description='github scrapping')
    parser.add_argument('-o', '--output', required=True,
                        help='for output log')
    parser.add_argument('-r', '--run_id', required=True,
                        help='run_id for db')
    parser.add_argument('-t', '--token', required=False,
                        help='token for github')
    parser.add_argument('-db', '--database', required=False, default='OSCP_data.db',
                        help='database for all data')

    args = parser.parse_args(cpr)
    root = args.output

    #  Logging managment
    log_file = os.path.join(root, fr'github_dl_id{args.run_id}_{now}.log')
    set_logger(log_file)
    log = logging.getLogger('main')
    log.critical("start")

    #  connect to db
    conn = init_db(args.database)

    #  Define path for input/output file
    repo_missing_path =  os.path.join(root, r'repos_name_missing.txt')

    ### Main process
    health_check = CheckpointManager(conn=conn)
    g = Github(login_or_token=args.token)  # init github conenction

    repos = get_todo_repos(conn, run_id=args.run_id)
    while len(repos) > 0:
        get_data(repos=repos, repo_missing_path=repo_missing_path, g=g, conn=conn, health_check=health_check)
        log.info(f"Batch of repos done, trying to take more")
        repos = get_todo_repos(conn, run_id=args.run_id)

    conn.close()
    log.info("end")

       
if __name__ == "__main__":
    #args = r"-o .\output -r 0 -db C:\Users\Thibaut\Documents\These\code\OSCP_data.db"
    main() #args.split(' '))