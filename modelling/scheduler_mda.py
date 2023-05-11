import pandas as pd
import pyomo.environ as pe
import pyomo.gdp as pyogdp
import re
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator
from itertools import product

class TheatreScheduler:
    def __init__(self, case_file_path, session_file_path):
        """
        Read case and session data into Pandas DataFrames
        Args:
            case_file_path (str): path to case data in CSV format
        """
        try:
            self.df_cases = pd.read_csv(case_file_path)
            self.df_cases_g = pd.DataFrame()
        except FileNotFoundError:
            print("Case data not found.")
        self.df_cases["Case Expected Date"].fillna(method='ffill', inplace = True)
        self.df_cases['Scheduled Setup Start']=pd.to_datetime(self.df_cases['Scheduled Setup Start'])
        self.df_cases['Scheduled Cleanup Complete']=pd.to_datetime(self.df_cases['Scheduled Cleanup Complete'])
        self.df_cases['Scheduled Room Duration']=(self.df_cases['Scheduled Cleanup Complete']-self.df_cases['Scheduled Setup Start']).astype('timedelta64[m]')
        self.df_cases['Room']=self.df_cases.apply(lambda row: int(re.search(r'\d+', row['Room']).group()), axis = 1)
        self.df_cases = self.df_cases.loc[self.df_cases['Room']<41]
#         self.df_cases = self.df_cases.loc[self.df_cases['Room']<7]
        self.df_cases = self.df_cases.dropna()
        self.df_cases['Day']=self.df_cases.groupby('Date').ngroup()
        self.df_cases['Room']=self.df_cases['Room']+self.df_cases['Day']*40
        self.df_cases['SurgeonID'] = self.df_cases.groupby('Primary Surgeon Name').ngroup()
        self.df_cases['SurgeonID'] = self.df_cases['SurgeonID'] + self.df_cases['Day']*40
        self.df_cases=self.df_cases.assign(CaseID=range(len(self.df_cases)))
        
        self.df_cases_g = self.df_cases.groupby('Primary Surgeon Name')['Scheduled Room Duration'].sum().reset_index()
        self.df_cases_g['CaseID'] = self.df_cases.groupby('Primary Surgeon Name')['CaseID'].min().reset_index()['CaseID'].to_dict()
        self.df_cases_g['SurgeonID'] = self.df_cases_g['CaseID']
        try:
            self.df_sessions = pd.read_csv(session_file_path)
#             self.df_sessions = self.df_sessions.loc[self.df_sessions['SessionID']<7]
        except FileNotFoundError:
            print("Session data not found")
        self.model = self.create_model()
    
    def _generate_case_durations(self):
        """
        Generate mapping of cases IDs to median case time for the procedure
        Returns:
            (dict): dictionary with CaseID as key and median case time (mins) for procedure as value
        """
        return pd.Series(self.df_cases["Scheduled Room Duration"].values, index=self.df_cases["CaseID"]).to_dict()
#         return pd.Series(self.df_cases_g["Scheduled Room Duration"].values, index=self.df_cases_g["CaseID"]).to_dict()

    def _generate_session_durations(self):
        """
        Generate mapping of all theatre sessions IDs to session duration in minutes
        Returns:
            (dict): dictionary with SessionID as key and session duration as value
        """
        return pd.Series(self.df_sessions["Duration"].values, index=self.df_sessions["SessionID"]).to_dict()

    def _generate_session_start_times(self):
        """
        Generate mapping from SessionID to session start time
        Returns:
            (dict): dictionary with SessionID as key and start time in minutes since midnight as value
        """
        # Convert session start time from HH:MM:SS format into seconds elapsed since midnight
        self.df_sessions.loc[:, "Start"] = pd.to_timedelta(self.df_sessions["Start"])
        self.df_sessions.loc[:, "Start"] = self.df_sessions["Start"].dt.total_seconds() / 60
        return pd.Series(self.df_sessions["Start"].values, index=self.df_sessions["SessionID"]).to_dict()

    def _get_ordinal_case_deadlines(self):
        """
        #TODO
        Returns:

        """
        self.df_cases.loc[:, "Case Expected Date"] = pd.to_datetime(self.df_cases["Case Expected Date"])
        self.df_cases.loc[:, "Case Expected Date"] = self.df_cases["Case Expected Date"].apply(lambda date: date.toordinal())
        return pd.Series(self.df_cases["Case Expected Date"].values, index=self.df_cases["CaseID"]).to_dict()
#         return pd.Series(self.df_cases["Case Expected Date"].values, index=self.df_cases_g["CaseID"]).to_dict()
    
    def _get_case_surgeon(self):
        """
        #TODO
        Returns:

        """
        self.df_cases.loc[:, "SurgeonID"] = self.df_cases_g["SurgeonID"]
        return pd.Series(self.df_cases_g["SurgeonID"].values, index=self.df_cases_g["CaseID"]).to_dict()
    
    def _get_ordinal_session_dates(self):
        """
        #TODO
        Returns:

        """
        self.df_sessions.loc[:, "Date"] = pd.to_datetime(self.df_sessions["Date"])
        self.df_sessions.loc[:, "Date"] = self.df_sessions["Date"].apply(lambda date: date.toordinal())
        return pd.Series(self.df_sessions["Date"].values, index=self.df_sessions["SessionID"]).to_dict()

    def _generate_disjunctions(self):
        """
        #TODO
        Returns:
            disjunctions (list): list of tuples containing disjunctions
        """
        cases = self.df_cases["CaseID"].to_list()
#         cases = self.df_cases_g["CaseID"].to_list()
        sessions = self.df_sessions["SessionID"].to_list()
        disjunctions = []
        for (case1, case2, session) in product(cases, cases, sessions):
            if (case1 != case2) and (case2, case1, session) not in disjunctions:
                disjunctions.append((case1, case2, session))

        return disjunctions
    
    def _generate_disjunctions_sur(self):
        """
        #TODO
        Returns:
            disjunctions (list): list of tuples containing disjunctions
        """
        cases = self.df_cases["CaseID"].to_list()
#         cases = self.df_cases_g["CaseID"].to_list()
        sessions = self.df_sessions["SessionID"].to_list()
        
        disjunctions = []
        for (case1, case2, session1, session2) in product(cases, cases, sessions, sessions):
            if (case1 != case2) and (session1 != session2) and (case2, case1, session2, session1) not in disjunctions \
            and (case2, case1, session1, session2) not in disjunctions \
             and \
            self.df_cases[self.df_cases["CaseID"] == case1]["SurgeonID"].item()\
            ==self.df_cases[self.df_cases["CaseID"] == case2]["SurgeonID"].item():
                disjunctions.append((case1, case2, session1, session2))

        return disjunctions
    
    def create_model(self):
        model = pe.ConcreteModel()

        # Model Data

        # List of case IDs in surgical waiting list
        model.CASES = pe.Set(initialize=self.df_cases["CaseID"].tolist())
#         model.CASES = pe.Set(initialize=self.df_cases_g["CaseID"].tolist())
        # List of sessions IDs
        model.SESSIONS = pe.Set(initialize=self.df_sessions["SessionID"].tolist())
        # List of surgeon IDs
        model.SURGEONS = pe.Set(initialize=self.df_cases["SurgeonID"].tolist())
        # List of tasks - all possible (caseID, sessionID) combination
        model.TASKS = pe.Set(initialize=model.CASES * model.SESSIONS, dimen=2)
        # The duration (median case time) for each operation
        model.CASE_DURATION = pe.Param(model.CASES, initialize=self._generate_case_durations())
        # The duration of each theatre session
        model.SESSION_DURATION = pe.Param(model.SESSIONS, initialize=self._generate_session_durations())
        # The start time of each theatre session
        model.SESSION_START_TIME = pe.Param(model.SESSIONS, initialize=self._generate_session_start_times())
        # The deadline of each case
        model.CASE_DEADLINES = pe.Param(model.CASES, initialize=self._get_ordinal_case_deadlines())
        # The date of each theatre session
        model.SESSION_DATES = pe.Param(model.SESSIONS, initialize=self._get_ordinal_session_dates())

        model.DISJUNCTIONS = pe.Set(initialize=self._generate_disjunctions(), dimen=3)
        model.DISJUNCTIONS_SUR = pe.Set(initialize=self._generate_disjunctions_sur(), dimen=4)

        ub = 1440  # minutes in a day
        model.M = pe.Param(initialize=1e3*ub)  # big M
        max_util = 1
        num_cases = self.df_cases.shape[0]

        # Decision Variables
        model.SESSION_ASSIGNED = pe.Var(model.TASKS, domain=pe.Binary)
        model.CASE_START_TIME = pe.Var(model.TASKS, bounds=(0, ub), within=pe.PositiveReals)
        model.CASES_IN_SESSION = pe.Var(model.SESSIONS, bounds=(0, num_cases), within=pe.PositiveReals)

        # Objective
        def objective_function(model):
            return pe.summation(model.CASES_IN_SESSION)
#             return sum([model.SESSION_ASSIGNED[case, session] for case in model.CASES for session in model.SESSIONS])
        model.OBJECTIVE = pe.Objective(rule=objective_function, sense=pe.maximize)

        # Constraints

        # Case start time must be after start time of assigned theatre session
        def case_start_time(model, case, session):
            return model.CASE_START_TIME[case, session] >= model.SESSION_START_TIME[session] - \
                   ((1 - model.SESSION_ASSIGNED[(case, session)])*model.M)
        model.CASE_START = pe.Constraint(model.TASKS, rule=case_start_time)

        # Case end time must be before end time of assigned theatre session
        def case_end_time(model, case, session):
            return model.CASE_START_TIME[case, session] + model.CASE_DURATION[case] <= model.SESSION_START_TIME[session] + \
                   model.SESSION_DURATION[session]*max_util + ((1 - model.SESSION_ASSIGNED[(case, session)]) * model.M)
        model.CASE_END_TIME = pe.Constraint(model.TASKS, rule=case_end_time)

        # Cases can be assigned to a maximum of one session
        def session_assignment(model, case):
            return sum([model.SESSION_ASSIGNED[(case, session)] for session in model.SESSIONS]) <= 1
        model.SESSION_ASSIGNMENT = pe.Constraint(model.CASES, rule=session_assignment)

        def set_deadline_condition(model, case, session):
            return model.SESSION_DATES[session] <= model.CASE_DEADLINES[case] + ((1 - model.SESSION_ASSIGNED[case, session])*model.M)
        model.APPLY_DEADLINE = pe.Constraint(model.TASKS, rule=set_deadline_condition)
        
        def no_surg_overlap(model, case1, case2, session1, session2):
            return [model.CASE_START_TIME[case1, session1] + model.CASE_DURATION[case1] <= model.CASE_START_TIME[case2, session2] + \
                    ((2 - model.SESSION_ASSIGNED[case1, session1] - model.SESSION_ASSIGNED[case2, session2])*model.M),
                    model.CASE_START_TIME[case2, session2] + model.CASE_DURATION[case2] <= model.CASE_START_TIME[case1, session1] + \
                    ((2 - model.SESSION_ASSIGNED[case1, session1] - model.SESSION_ASSIGNED[case2, session2])*model.M)]
            
        model.DISJUNCTIONS_RULE_SUR = pyogdp.Disjunction(model.DISJUNCTIONS_SUR, rule=no_surg_overlap)
        
        def no_case_overlap(model, case1, case2, session):
            return [model.CASE_START_TIME[case1, session] + model.CASE_DURATION[case1] <= model.CASE_START_TIME[case2, session] + \
                    ((2 - model.SESSION_ASSIGNED[case1, session] - model.SESSION_ASSIGNED[case2, session])*model.M),
                    model.CASE_START_TIME[case2, session] + model.CASE_DURATION[case2] <= model.CASE_START_TIME[case1, session] + \
                    ((2 - model.SESSION_ASSIGNED[case1, session] - model.SESSION_ASSIGNED[case2, session])*model.M)]
        model.DISJUNCTIONS_RULE = pyogdp.Disjunction(model.DISJUNCTIONS, rule=no_case_overlap)

        def theatre_util(model, session):
            return model.CASES_IN_SESSION[session] == \
                   sum([model.SESSION_ASSIGNED[case, session] for case in model.CASES])

        model.THEATRE_UTIL = pe.Constraint(model.SESSIONS, rule=theatre_util)

        pe.TransformationFactory("gdp.bigm").apply_to(model)

        return model
    
    def solve(self, solver_name, options=None, solver_path=None, local=True):

        if solver_path is not None:
            solver = pe.SolverFactory(solver_name, executable=solver_path)
        else:
            solver = pe.SolverFactory(solver_name)

        # TODO remove - too similar to alstom
        if options is not None:
            for key, value in options.items():
                solver.options[key] = value

        if local:
            solver_results = solver.solve(self.model, tee=True, warmstart=True)
        else:
            solver_manager = pe.SolverManagerFactory("neos")
            solver_results = solver_manager.solve(self.model, opt=solver,  warmstart=True)
        iter_num=solver_results['Solver'][0]['Statistics']['Black box']['Number of iterations']
        results = [{"Case": case,
                    "Session": session,
                    "Session Date": self.model.SESSION_DATES[session],
                    "Case Deadline": self.model.CASE_DEADLINES[case],
                    "Days before deadline": self.model.CASE_DEADLINES[case] - self.model.SESSION_DATES[session],
                    "Start": self.model.CASE_START_TIME[case, session](),
                    "Assignment": self.model.SESSION_ASSIGNED[case, session]()}
                   for (case, session) in self.model.TASKS]

        self.df_times = pd.DataFrame(results)

        all_cases = self.model.CASES.value_list
        cases_assigned = []
        for (case, session) in self.model.SESSION_ASSIGNED:
            if self.model.SESSION_ASSIGNED[case, session] == 1:
                cases_assigned.append(case)

        cases_missed = list(set(all_cases).difference(cases_assigned))
        print("Number of cases assigned = {} out of {}:".format(len(cases_assigned), len(all_cases)))
        print("Cases assigned: ", cases_assigned)
        print("Number of cases missed = {} out of {}:".format(len(cases_missed), len(all_cases)))
        print("Cases missed: ", cases_missed)
        self.model.CASES_IN_SESSION.pprint()
        print("Total Objective = {}".format(sum(self.model.CASES_IN_SESSION.get_values().values())))
        print("Number of constraints = {}".format(solver_results["Problem"].__getitem__(0)["Number of constraints"]))
        #self.model.SESSION_ASSIGNED.pprint()
        print(self.df_times[self.df_times["Assignment"] == 1].to_string())
        self.draw_gantt(iter_num)
#         self.draw_gantt_surgeon(iter_num)

    def draw_gantt(self,iter_num):
        df = self.df_times[self.df_times["Assignment"] == 1]
        cases = sorted(list(df['Case'].unique()))
        sessions = sorted(list(df['Session'].unique()))
        bar_style = {'alpha': 1.0, 'lw': 6, 'solid_capstyle': 'butt'}
        text_style = {'color': 'white', 'ha': 'center', 'va': 'center'}
        colors = cm.Dark2.colors

        df.sort_values(by=['Case', 'Session'])
        df.set_index(['Case', 'Session'], inplace=True)

        fig, ax = plt.subplots(1, 1)
        for c_ix, c in enumerate(cases, 1):
            for s_ix, s in enumerate(sessions, 1):
                if (c, s) in df.index:
                    xs = df.loc[(c, s), 'Start'] /60
                    xf = (df.loc[(c, s), 'Start'] + self.df_cases[self.df_cases["CaseID"] == c]["Scheduled Room Duration"])/60
                    ax.plot([xs, xf], [s] * 2, c=colors[self.df_cases[self.df_cases["CaseID"] == c]['SurgeonID'].item() % len(colors)], **bar_style)
                    ax.text((xs + xf) / 2, s, self.df_cases[self.df_cases["CaseID"] == c]["Primary Surgeon Name"].item()[:2], **text_style)

#         ax.set_title('Mays OR (Optimized)')
        ax.set_title('Iterations: '+str(iter_num))
        ax.set_xlabel('Time')
        ax.set_xlim([7, 24])
        ax.set_ylim([0, max(sessions)+1])
        ax.set_ylabel('OR')
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        fig.tight_layout()
        return fig
        
    def draw_gantt_surgeon(self,iter_num):
        df = self.df_times[self.df_times["Assignment"] == 1]
        cases = sorted(list(df['Case'].unique()))
        sessions = sorted(list(df['Session'].unique()))
        bar_style = {'alpha': 1.0, 'lw': 6, 'solid_capstyle': 'butt'}
        text_style = {'color': 'white', 'ha': 'center', 'va': 'center'}
        colors = cm.Dark2.colors

        df.sort_values(by=['Case', 'Session'])
        df.set_index(['Case', 'Session'], inplace=True)

        fig, ax = plt.subplots(1, 1)
        prev_f={}
        for c_ix, c in enumerate(cases, 1):
            for s_ix, s in enumerate(sessions, 1):
                if (c, s) in df.index:
                    if s in prev_f:
                        xs = prev_f[s]
                        xf = prev_f[s] + self.df_cases_g[self.df_cases_g["CaseID"] == c]["Scheduled Room Duration"].item()/60
                    else:
                        xs = 7
                        xf = 7 + self.df_cases_g[self.df_cases_g["CaseID"] == c]["Scheduled Room Duration"].item()/60
                    prev_f[s]=xf
                    
                    curr_cases = self.df_cases[self.df_cases["SurgeonID"] == self.df_cases_g[self.df_cases_g["CaseID"] == c]['SurgeonID'].item()].sort_values(by=['Scheduled Setup Start'])["Scheduled Room Duration"]
                    xf_curr=xs
                    xs_curr=xs
                    for i in curr_cases:
                        xf_curr+=i/60
                        ax.plot([xs_curr, xf_curr], [s] * 2, c=colors[self.df_cases_g[self.df_cases_g["CaseID"] == c]['SurgeonID'].item() % len(colors)], **bar_style)
                        ax.text((xs_curr + xf_curr) / 2, s, self.df_cases_g[self.df_cases_g["CaseID"] == c]["Primary Surgeon Name"].item()[:2], **text_style)
                        xs_curr+=i/60

#         ax.set_title('Mays OR (Optimized)')
        ax.set_title('Iterations: '+str(iter_num))
        ax.set_xlabel('Time')
        ax.set_xlim([7, 18.25])
        ax.set_ylabel('OR')
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        fig.tight_layout()
        return fig