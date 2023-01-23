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
import gc

from github import Github

from sizeof import total_size, sizeof_fmt

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
        gc.collect()  # clear memory


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

    def health_check(self, g, ev=None, moment='', force_save=True):
        
        log = logging.getLogger('main')
        if isinstance(ev, Event):
            df = ev.to_dataframe()
            if (len(df) > 2) or force_save:
                log.warning(f'Cleaning  ev at {moment}')
                self.date_checkpoint(df, moment)
                ev.clean_for_savepoint()
            # clear memory
            del df
            gc.collect()

        self.check_remaining_request(g)
        return ev


    def date_checkpoint(self, df, moment):
        """
        Checkpoint to save data in the DB because of raw overflow
        """
        log = logging.getLogger('main')

        log.debug(f'Data checkpoint at {moment} for {df.repo.unique()}')
        count = df.to_sql(name='event', con=self.conn, index=False, if_exists = 'append', dtype='string')
        log.debug(f"{len(df)} rows added to event (count from sql={count}")
        self.event_ids.union(set(df.id.to_list()))

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
                log.critical(f"Problem with duration {duration}")
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

def get_event_from_pr(pr, repo, g, health_check, conn, based_ev=None):
    """
    Explore a PullRequest to find all contributor
    # TODO what about commits ?
    """
    health_check.health_check(g=g, moment='Get event from pr start')
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

    
    health_check.health_check(g=g, moment='Get event from pr before participant')
    log_memory(locals())

    try:
        ev.add_participants(get_from_named_user(pr.user), contrib_type=ContribType.dev)
    except Exception as err:
                    log.critical('PR get pr user')
                    error_log(err=err, conn=conn, repo=repo, type='pr')

    ev = health_check.health_check(g=g, ev=ev, moment='PR after participant')
    log_memory(locals())

    try:
        assignes = tuple(get_from_named_user(user) for user in pr.assignees)
        if assignes:
            ev.add_participants(assignes, contrib_type=ContribType.dev)
    except Exception as err:
        log.critical('PR get assigne')
        error_log(err=err, conn=conn, repo=repo, type='pr')

    ev = health_check.health_check(ev=ev, g=g, moment='GET PR before comment')
    log_memory(locals())

        # comments
    if pr.comments:
        log.debug(f"{pr.comments} comments found")
        for c in pr.get_comments():
            try:
                if c:
                    ev.add_participants(get_from_named_user(c.user), contrib_type=ContribType.comment)
                    ev = health_check.health_check(ev=ev, g=g, moment='Get event from pr during comment')
                else:
                    log.warning(f"contributor missing in comments")
            except Exception as err:
                log.critical('PR get comment')
                error_log(err=err, conn=conn, repo=repo, type='pr')
    
    log_memory(locals())
    ev = health_check.health_check(ev=ev, g=g, moment='Get event from pr before review comment')
    if pr.review_comments:
        log.debug(f"{pr.review_comments} review_comments found")
        for c in pr.get_review_comments():
            try:
                if c:
                    ev.add_participants(get_from_named_user(c.user), contrib_type=ContribType.comment)
                    ev = health_check.health_check(ev=ev, g=g, moment='Get event from pr during review comment')
                else:
                    log.warning(f"contributor missing in review_comment")
            except Exception as err:
                log.critical('PR get')
                error_log(err=err, conn=conn, repo=repo, type='pr')
   
    log_memory(locals())
    ev = health_check.health_check(ev=ev, g=g, moment='Get event from pr before issue comment')
    com = pr.get_issue_comments()
    if com.totalCount:
        log.debug(f"{com.totalCount} issue comments found")
        for c in com:
            try:
                if c:
                    ev.add_participants(get_from_named_user(c.user), contrib_type=ContribType.comment)
                    ev = health_check.health_check(ev=ev, g=g, moment='Get event from pr during issue comment')
                else:
                    log.warning(f"contributor missing in issue comment")
            except Exception as err:
                log.critical('PR get')
                error_log(err=err, conn=conn, repo=repo, type='pr')
    del com
    gc.collect()

    # commits
    ev = health_check.health_check(ev=ev, g=g, moment='Get event from pr before commits')
    if pr.commits:
        log.debug(f"{pr.commits} commits")
        for c in pr.get_commits():
            try:
                if c.author:
                    ev.add_participants(get_from_named_user(c.author), contrib_type=ContribType.dev)
                elif c.commit.author:
                    ev.add_participants(get_from_named_user(c.commit.author), contrib_type=ContribType.dev)
                else:
                    log.warning(f"contributor missing in commit")
                ev = health_check.health_check(ev=ev, g=g, moment='Get event from pr during commit')
            except Exception as err:
                log.critical('PR get')
                error_log(err=err, conn=conn, repo=repo, type='pr')

    ev = health_check.health_check(ev=ev, g=g, moment='Get event from pr', force_save=True)
    return health_check


def error_log(err, conn, repo, type):
    """
    Log error with stack trace
    """
    log = logging.getLogger('main')


    query = f"UPDATE repo SET done = {-1}, ND = \"{type}\" WHERE name = \"{repo}\""
    log.debug(f"Query update : {query}")
    cursor = conn.cursor()
    cursor.execute(query)
    conn.commit()
    cursor.close()
    log.debug(f"{repo} finished")

    log.error(err, exc_info=True)

def get_todo_repos(conn, run_id):
    limit = 1
    log = logging.getLogger("main")
    # Get list of repo
    cursor = conn.cursor()
    log.debug("choose repos")
    q = f"SELECT count(distinct name) FROM repo WHERE token_id = {run_id} AND done = {0} LIMIT {limit}"
    cursor.execute(q)
    result = cursor.fetchall()
    if result[0][0] == 0 :
        q = f"UPDATE repo SET token_id={run_id} WHERE id IN (SELECT id FROM repo WHERE token_id=-1 LIMIT {limit})"
        cursor.execute(q)
        conn.commit()

    query = f"SELECT name FROM repo WHERE token_id = {run_id} AND done = {0} LIMIT {limit}"
    repos = [ret[0] for ret in cursor.execute(query).fetchall()]
    cursor.close()
    log.debug(f"find {len(repos)} to do")

    return repos

def log_memory(var: dict):
    total = 0
    for k,v in var.items():
        s = total_size(v)
        if s > 10**9:  # 1Go
            logging.critical(f"MEMORY ALERT : {k} at {sizeof_fmt(s)}")
        elif s > 10**8:  # 100Mo
            logging.warning(f"Memory alert : {k} at {sizeof_fmt(s)}")
        total += s
    if total > 7*10**8:  # 700Mo
        logging.critical(f"MEMORY ALERT over all vars {sizeof_fmt(total)}")
        for k,v in var.items():
            logging.critical(f"{v} : {sizeof_fmt(total)}")

def get_pr_info(pr, repo, g, health_check, conn):
    """
    Get data about PR (created, closed ...) one line by pr for DB
    """
    health_check.health_check(g=g, moment='Get event info for PR')

    attribut = ['id', 'number', 'created_at', 'closed_at', 'merged', 'merged_at', 'state']
    if pr.as_issue():
        typ = EventType.ISSUEPR.value
    else:
        typ = EventType.ONLYPR.value
        
    df = pd.DataFrame({'repo': repo, 'type': typ}, index=[0])

    for a in attribut:
        df[a] = getattr(pr, a, pd.NA)
    
    user = get_from_named_user(pr.user)
    df_user = pd.DataFrame(data=[user], columns=['user_id', 'user_login', 'user_name', 'user_email'], index=[0])
    df = pd.concat([df, df_user], axis=1)


    df.to_sql('event', con=conn, if_exists='append', dtype='str', index=False)
    return health_check


def get_iss_info(iss, repo, g, health_check, conn):
    """
    Get data about issue (created, closed ...) one line by issue for DB
    
    """
    health_check.health_check(g=g, moment='Get event info for PR')

    attribut = ['id', 'number', 'created_at', 'closed_at', 'merged', 'merged_at', 'state']

    user = get_from_named_user(iss.user)
        
    df = pd.DataFrame({'repo': repo, 'type': EventType.ONLYISSUE.value}, index=[0])

    for a in attribut:
        df[a] = getattr(iss, a, pd.NA)
    
    df_user = pd.DataFrame(user, ['user_id', 'user_login', 'user_name', 'user_email'])
    df = pd.concat([df, df_user])
    df.to_sql('event', con=conn, if_exists='append', dtype='str', index=False)
    return health_check


def get_data(repos, g, conn, health_check):
    log = logging.getLogger("main")
    for repo in tqdm.tqdm(repos,desc="Repos", leave=True, position=0):  # iterate over all repos
        health_check.event_ids = set()

        #  Connect to repo
        try:
            log.info(f'Doing repo {repo}')
            health_check.health_check(g=g, moment='start_repo')
            rep = g.get_repo(repo)

        except Exception as err:
            error_log(err=err, conn=conn, repo=repo, type='get_repo')
            continue

        # Get all PR from the repo
        
        log.debug('Take pull request')
        health_check.health_check(g=g, moment='start PR')

        skip = 0
        try:
            prs = rep.get_pulls(state='all')  # get all pr
            log.debug(f"{prs.totalCount} prs found")
            log_memory(locals())

        except Exception as err:
            skip=1
            log.critical('PR')
            error_log(err=err, conn=conn, repo=repo, type='pr')

        if not skip:
            for pr in tqdm.tqdm(prs, total=prs.totalCount, desc="PR", leave=False, position=1):
                log_memory(locals())
                #health_check = get_event_from_pr(pr=pr, repo=repo, g=g, health_check=health_check, conn=conn)  # used to get commit and comment
                health_check = get_pr_info(pr=pr, repo=repo, g=g, health_check=health_check, conn=conn)  # used to get pr data
        
        del prs
        gc.collect()
        skip=0
        # Get all issues from the repo
        try:
            log.debug('Take issues')
            health_check.health_check(g=g, moment="Issue")
            issues = rep.get_issues(state='all')

        except Exception as err:
            log.critical('ISSUE')
            error_log(err=err, conn=conn, repo=repo, type='issue')
            continue

        log.debug(f"{issues.totalCount} issues found")
        for iss in tqdm.tqdm(issues,desc="Issue", total=issues.totalCount, leave=False, position=1):
            log_memory(locals())

            pr = iss.pull_request
            ev = Event(repo=repo, id=iss.id, nbr=iss.number)

            if pr:
                ev.etype = EventType.ISSUEPR.value
                health_check.health_check(g=g, moment='pr in issue')
                pr = iss.as_pull_request()
                if pr.id in health_check.event_ids:
                    continue
                else:
                    # health_check = get_event_from_pr(pr, repo, g, health_check=health_check, conn=conn, based_ev=ev)
                    health_check = get_pr_info(pr=pr, repo=repo, g=g, health_check=health_check, conn=conn)  # used to get pr data

            else:
                    health_check = get_iss_info(iss=iss, repo=repo, g=g, health_check=health_check, conn=conn)
            #     ev.etype = EventType.ONLYISSUE.value

            #     try:
            #         health_check.health_check(g, moment='issue not pr')
            #         ev.add_participants(get_from_named_user(iss.user), contrib_type=ContribType.comment)
            #     except Exception as err:
            #         log.critical('ISSUE')
            #         error_log(err=err, conn=conn, repo=repo, type='issue')
            #     log_memory(locals())

            #     try:
            #         ev = health_check.health_check(g=g, ev=ev, moment='issue user')
            #         assignes = tuple(get_from_named_user(user) for user in iss.assignees)
            #         if assignes:
            #             ev.add_participants(assignes, contrib_type=ContribType.comment)
            #     except Exception as err:
            #         log.critical('ISSUE')
            #         error_log(err=err, conn=conn, repo=repo, type='issue')

            #     try:    
            #         # comments
            #         ev = health_check.health_check(g=g, ev=ev, moment='issue comment')
            #         com = iss.get_comments()
            #         for c in com:
            #             log_memory(locals())
            #             ev.add_participants(get_from_named_user(c.user), contrib_type=ContribType.comment)
            #             ev = health_check.health_check(g=g, ev=ev, moment='issue comment')
            #         del com
            #     except Exception as err:
            #         log.critical('ISSUE')
            #         error_log(err=err, conn=conn, repo=repo, type='issue')
            # ev = health_check.health_check(g=g, ev=ev, moment='issue comment', force_save=True)
                
        del issues
        gc.collect()


        try:    
            log.debug(f"Ending {repo}")

            query = f"UPDATE repo SET done = {1} WHERE name = \"{repo}\""
            log.debug(f"Query update : {query}")
            cursor = conn.cursor()
            cursor.execute(query)
            conn.commit()
            cursor.close()
            log.debug(f"{repo} finished")
        
        except Exception as err:
            log.critical('END')
            error_log(err=err, conn=conn, repo=repo, type='end')

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

    ### Main process
    health_check = CheckpointManager(conn=conn)
    g = Github(login_or_token=args.token)  # init github conenction

    repos = get_todo_repos(conn, run_id=args.run_id)
    while len(repos) > 0:
        get_data(repos=repos, g=g, conn=conn, health_check=health_check)
        log.info(f"Batch of repos done, trying to take more")
        repos = get_todo_repos(conn, run_id=args.run_id)

    conn.close()
    log.info("end")

       
if __name__ == "__main__":
    args = r"-o .\output -r 0 -db C:\Users\Thibaut\Documents\These\code\OSCP_data.db"
    main(args.split(' '))