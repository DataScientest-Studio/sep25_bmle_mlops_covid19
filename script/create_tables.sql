CREATE TABLE parameters
(
    validity_date 				timestamp		PRIMARY KEY DEFAULT now(),
    retraining_trigger_ratio 	FLOAT 			CHECK (retraining_trigger_ratio >= 0 AND retraining_trigger_ratio <= 1),
	img_width					INTEGER			NOT NULL,
	img_height					INTEGER			NOT NULL,
	gray_mode					BOOLEAN			NOT NULL,
	batch_size					INTEGER			CHECK (batch_size > 0 AND (batch_size & (batch_size - 1)) = 0),
	train_size					FLOAT			CHECK (train_size >= 0 AND train_size <= 1),
	random_state				INTEGER			DEFAULT 42,
	optimizer_name				VARCHAR			NOT NULL,
	loss_cat					VARCHAR			NOT NULL,
	metrics						VARCHAR			NOT NULL,
	ES_patience					INTEGER			CHECK (ES_patience > 0),
	ES_min_delta				FLOAT			CHECK (ES_min_delta > 0),
	ES_mode						VARCHAR			NOT NULL,
	ES_monitor					VARCHAR			NOT NULL,
	RLROP_patience				VARCHAR			NOT NULL,
	RLROP_monitor				VARCHAR			NOT NULL,
	RLROP_min_delta				FLOAT			CHECK (RLROP_min_delta > 0),
	RLROP_factor				FLOAT			CHECK (RLROP_factor > 0),
	RLROP_cooldown				INTEGER			CHECK (RLROP_cooldown > 0),
	nb_layer_to_freeze			INTEGER			CHECK (nb_layer_to_freeze >= 0)
);

CREATE TABLE training_log
(
  training_date 				timestamp		PRIMARY KEY ,
	modification_date			timestamp NOT NULL,
	run_id							varchar			NOT NULL,
	model_name					varchar			NOT NULL, 
	stage							varchar 		not null,
	training_size				INTEGER			CHECK (training_size >= 0),
	validation_size				INTEGER			CHECK (validation_size >= 0),
	epochs_number				INTEGER			CHECK (epochs_number >= 0),
    accuracy					FLOAT			CHECK (accuracy >= 0 AND accuracy <= 1),
	class_0_precision			FLOAT			CHECK (class_0_precision >= 0 AND class_0_precision <= 1),
	class_0_recall				FLOAT			CHECK (class_0_recall >= 0 AND class_0_recall <= 1),
	class_0_f1					FLOAT			CHECK (class_0_f1 >= 0 AND class_0_f1 <= 1),
	class_1_precision			FLOAT			CHECK (class_1_precision >= 0 AND class_1_precision <= 1),
	class_1_recall				FLOAT			CHECK (class_1_recall >= 0 AND class_1_recall <= 1),
	class_1_f1					FLOAT			CHECK (class_1_f1 >= 0 AND class_1_f1 <= 1),
	true_class_0				INTEGER			CHECK (true_class_0 >= 0 AND true_class_0 <= validation_size),
	false_class_0				INTEGER			CHECK (false_class_0 >= 0 AND true_class_0 <= validation_size),
	true_class_1				INTEGER			CHECK (true_class_1 >= 0 AND true_class_0 <= validation_size),
	false_class_1				INTEGER			CHECK (false_class_1 >= 0 AND true_class_0 <= validation_size)
);

CREATE TABLE images_dataset
(
	id 				SERIAL 		PRIMARY KEY,
    image_url 	varchar 		NOT NULL,
    mask_url  	varchar,
	class_type		CHAR		CHECK (class_type = '0' OR class_type = '1'),
    injection_date 	TIMESTAMP 	NOT NULL,
    created_at     	TIMESTAMP 	DEFAULT NOW()
);

CREATE TABLE feedback
(
		id 							SERIAL 		PRIMARY KEY,
		img_id					integer		REFERENCES  images_dataset(id),
		feedback_date		TIMESTAMP DEFAULT NOW(),
		predicted_class	char			NOT NULL,
		diagnostic			char			NOT NULL,
		comment					varchar
)