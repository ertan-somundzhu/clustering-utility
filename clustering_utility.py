import pandas
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage, fclusterdata
from statsmodels.stats.oneway import anova_oneway



class itIsTimeToHierarchicallyClusterSomething:

    """
    this class is intended for simplifying hierarchical clustering

    it is capable of doing the following things:
        1) it carries clustering with specified parameters 
        2) draws the dendogram
        3) returns clustering labels of specified levels

    parameters:
        data: pandas.DataFrame - data
        X: list[str] - list of variables on which the clustering will be carried out
        method: str - the linkage method to use
        metric: str - the distance metric for calculating pairwise distances
        num_of_clusters: list[int] - list of different numbers of clusters you may want to check
        levels_of_dendogram: int - depth of dendogram visualisation (do not set a high number, unless you have powerful computer)
        return_table: bool - if True returns the original data with columns containing clustering labels, if False just returns columns containing clustering labels

    Note: before using draw_the_dendogram() and draw_the_dendogram() run do_some_clustering_please(),
    otherwise an error will be raised

    """

    def __init__(
            self, 
            data: pandas.DataFrame, 
            X: list[str], 
            method: str, 
            metric: str, 
            num_of_clusters: list[int],
            levels_of_dendogram: int,
            return_table: bool
        ):

        self.data = data
        self.X = X
        self.method = method
        self.metric = metric
        self.n = num_of_clusters
        self.p = levels_of_dendogram
        self.use_flag = False
        self.return_table = return_table

    def do_some_clustering_please(self):

        """
        this function is for carrying out hierarchical clustering.

        1) linkage is for creating a dendogram
        2) fclusterdata is for needed for class labels
        """

        # here several clustering models are created
        # linkage will be used showing the dendogram
        # fclusterdata will be created with differnt number of clusters and
        # later be used for assigning cluster labels to each observation
        clu_list = []

        self.Z= linkage(self.data[self.X], method=self.method, metric=self.metric)

        for i in self.n:
            clu = fclusterdata(self.data[self.X], t=i, criterion="maxclust", method=self.method, metric=self.metric)
            clu_list.append((i, clu))
            clu = None

        self.clu_list = clu_list

        # somewhat of a memory cleaning, i think
        clu_list = None
        self.use_flag = True

    def draw_the_dendogram(self):

        """
        this function draws the dendogram, that's it
        """
        if self.use_flag:
            plt.figure()
            dendrogram(self.Z, p=self.p, truncate_mode="level", no_labels=True)
            plt.title(self.method + " + " + self.metric)
            plt.show()
        else:
            raise ValueError("please, use do_some_clustering_please() first, then try to visualise the dendogram")

    def assign_labels(self):

        """
        this function returns cluster labels
        """

        if self.use_flag:

            clust_label_df = pandas.DataFrame()

            for i, clu in self.clu_list:

                col_name = f"h_clust_{i}_{self.method}_{self.metric}"

                clust_label_df[col_name] = clu

            if self.return_table:
                data = self.data[self.X].copy()
                data[clust_label_df.columns] = clust_label_df
                return data
            else:
                return clust_label_df
        else:
            raise ValueError("please, use do_some_clustering_please() first, then try to assign clustering labels")


class hierarchicalClusteringParameterTesting:

    """
    this class is intended for simplifying the testing of hierarchical clustering

    it is capable of doing the following things:
        1) gridsearches and selects models based on given limits 
        2) tests the contrast of clusters of models selected during the previous step via oneway anova
        3) returns coloured tables for cluster interpretaion of selcted models

    parameters:
        data: pandas.DataFrame - data
        X: list[str] - list of variables on which the clustering will be carried out
        methods: list[str] - the linkage methods to test
        metrics: list[str] - the distance metrics for calculating pairwise distances that need to be tested
        num_of_clusters: list[int] - list of different numbers of clusters you may want to check
        levels_of_dendogram: int - depth of dendogram visualisation (do not set a high number, unless you have powerful computer)
        return_table: bool - if True returns the original data with columns containing clustering labels, if False just returns columns containing clustering labels
        upper_limit: float - the maximum percentage of observations a cluster can take up
        lower_limit: float - the minimim percentage of observations a cluster can take up

    Note: before using draw_the_dendogram() and draw_the_dendogram() run do_some_clustering_please(),
    otherwise an error will be raised

    """

    def __init__(
            self, 
            upper_limit: float, 
            lower_limit: float, 
            levels_of_dendogram: int,
            methods: list[str],
            metrics: list[str],
            num_of_clusters: list[int],
            data: pandas.DataFrame,
            X: list[str]
        ):

        self.upper_limit = upper_limit
        self.lower_limit = lower_limit
        self.levels_of_dendogram = levels_of_dendogram
        self.methods = methods
        self.metrics = metrics
        self.num_of_clusters = num_of_clusters
        self.data = data
        self.X = X

    def test_fullness(self):

        """
        this function tests fullness of different clustering models
        """

        test_res_df = self.data[self.X].copy()
        shape = test_res_df.shape[0]
        upper_limit = self.upper_limit
        lower_limit = self.lower_limit

        # here differnt combinations of metrics and linkage methods are tried out
        for method in self.methods:
            for metric in self.metrics:

                try:
                    clust_model = itIsTimeToHierarchicallyClusterSomething(
                        data=self.data,
                        X=self.X,
                        method=method,
                        metric=metric,
                        num_of_clusters=self.num_of_clusters,
                        levels_of_dendogram=self.levels_of_dendogram,
                        return_table=False
                    )

                    clust_model.do_some_clustering_please()

                    loop_res = clust_model.assign_labels() 

                    for col in loop_res.columns:

                        # here the percentage of how many observations were put into one cluster
                        # if the number is higher or lower than the limit for ANY cluster
                        # then the model is thrown out
                        loop_res_valid = loop_res[col].value_counts() / shape
                        loop_res_valid = (upper_limit > loop_res_valid) & (loop_res_valid >= lower_limit)

                        if loop_res_valid.all():
                            test_res_df[col] = loop_res[col]
                        else:
                            pass

                except:
                    print(f"this combination (method = {method}, metric = {metric}) raised an error")
        
        self.test_res_df = test_res_df

        return test_res_df
    
    
    def test_contrast(self):

        """
        this function test contrast between clusters via oneway anove
        """

        test_res_df = self.test_res_df
        cols = self.X
        rows = test_res_df.drop(cols, axis=1).columns

        pvalue_df = pandas.DataFrame(
            columns=cols,
            index=rows 
        )

        # here we test if the differnce between means of different variables in different clusters id statisticall significant
        for col in cols:
            for row in rows:

                pvalue = anova_oneway(
                            data=test_res_df[col],
                            groups=test_res_df[row]
                        ).pvalue
                
                pvalue_df.loc[row, col] = pvalue

        def _color_red_or_green(val):
            color = "red" if val > 0.05 else "green"
            return "color: %s" % color

        return pvalue_df.style.map(_color_red_or_green)
    
    def interpret(self):

        """
        this function colours and prints out interpretation tables
        """

        test_res_df = self.test_res_df
        cols = self.X
        clust = test_res_df.drop(cols, axis=1).columns
        
        df_dict = {}

        for c in clust:

            labels = list(set(test_res_df[c].to_list()))
            index = [f"cluster_{i}" for i in labels]
            df = pandas.DataFrame(
                columns=cols,
                index=index
            )

            for col in cols:
                for label in labels:
                    df.loc[f"cluster_{label}", col] = test_res_df[test_res_df[c] == label][col].mean()
            
            df_dict[c] = df.style.background_gradient(cmap="RdYlGn", subset=cols)
        
        return df_dict