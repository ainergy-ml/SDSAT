
class Scorer(object):

    def __init__(self, name, sat):
        self.name = name
        self.sat = sat
        self.max_new_tokens = None
        self.i = 0
    
        # pred & ref values
        self.predictions = []
        self.references = []
        self.input_ids = []
        self.output_ids = []

        # benchmark values
        self.loop_times = []
        self.accept_rates = []
        self.tokens_per_s = []
        self.iters = []
        self.tot_mean_time = 0
        self.tot_mean_iter = 0
        self.total_input = 0
        self.total_output = 0

        # inline values
        self.current_pred = None
        self.current_time = None
        self.current_iter = None

    def update_metrics(self, time, input_ids, iter, accept_rate, avg_loop_time, 
                       token_per_s, output_ids, output_strs, gold, max_new_tokens):
        self.max_new_tokens = max_new_tokens
        self.loop_times.append(avg_loop_time)
        self.accept_rates.append(accept_rate)
        self.tokens_per_s.append(token_per_s)
        self.iters.append(iter)
        self.tot_mean_time += (time - self.tot_mean_time) / (self.i + 1)
        self.tot_mean_iter += (iter - self.tot_mean_iter) / (self.i + 1)
        self.total_input += input_ids.shape[1]
        self.total_output += output_ids.shape[1]

        self.input_ids.append(input_ids)
        self.output_ids.append(output_ids)
        self.predictions.append(output_strs)
        self.references.append([gold])

        self.current_pred = output_strs
        self.current_time = time
        self.current_iter = iter

        self.i += 1
