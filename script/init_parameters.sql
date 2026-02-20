INSERT INTO parameters(
	validity_date, retraining_trigger_ratio, img_width, img_height, gray_mode, batch_size, train_size, 
	random_state, optimizer_name, loss_cat, metrics, 
	es_patience, es_min_delta, es_mode, es_monitor, 
	rlrop_patience, rlrop_monitor, rlrop_min_delta, rlrop_factor, rlrop_cooldown, 
	nb_layer_to_freeze)
	VALUES (now(), 0.7, 299, 299, False, 32, 0.8, 
			42, 'adam', 'categorical_crossentropy', 'accuracy',
			5, 0.01, 'min', 'loss', 
			3, 'loss', 0.01, 0.1, 4, 
			0);