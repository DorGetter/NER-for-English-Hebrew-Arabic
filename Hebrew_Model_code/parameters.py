from datetime import datetime

# ========================================
#               Parameters
# ========================================

batch_size = 32
epochs = 25
max_sequence_length = 75
learning_rate = 3e-5
optimizer_tolerance = 1e-8
optimizer_weight_decay_rate = 0.01
maximal_gradients_norm = 1.0
scheduler_warmup_steps = 0
scheduler_end_lr_factor = 0.5
todays_date_time = datetime.now().strftime("%d:%m_%H:%M:%S")
