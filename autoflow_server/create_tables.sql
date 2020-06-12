create database autoflow;
\c autoflow;
create table dataset
(
    dataset_id          char(32)    not null,
    user_id             integer     not null,
    dataset_metadata    json        not null,
    dataset_path        text        not null,
    upload_type         varchar(32) not null,
    dataset_type        varchar(32) not null,
    dataset_source      varchar(32) not null,
    column_descriptions json        not null,
    columns_mapper      json        not null,
    columns             json        not null,
    create_time         timestamp   not null,
    modify_time         timestamp   not null,
    constraint dataset_pkey
        primary key (dataset_id, user_id)
);
create table experiment
(
    experiment_id     serial       not null
        constraint experiment_pkey
            primary key,
    user_id           integer      not null,
    hdl_id            char(32)     not null,
    task_id           char(32)     not null,
    experiment_type   varchar(128) not null,
    experiment_config json         not null,
    additional_info   json         not null,
    final_model_path  text,
    log_path          text,
    start_time        timestamp    not null,
    end_time          timestamp
);
create table hdl
(
    task_id      char(32)  not null,
    hdl_id       char(32)  not null,
    user_id      integer   not null,
    hdl          json      not null,
    hdl_metadata json      not null,
    create_time  timestamp not null,
    modify_time  timestamp not null,
    constraint hdl_pkey
        primary key (task_id, hdl_id, user_id)
);
create table run_history
(
    run_id          char(256)   not null
        constraint run_history_pkey
            primary key,
    config_id       char(128)   not null,
    config          json        not null,
    config_origin   varchar(64) not null,
    cost            real        not null,
    time            real        not null,
    instance_id     char(128)   not null,
    seed            integer     not null,
    status          integer     not null,
    additional_info json        not null,
    origin          integer     not null,
    pid             integer     not null,
    create_time     timestamp   not null,
    modify_time     timestamp   not null
);
create index run_history_instance_id
    on run_history (instance_id);

create table task
(
    task_id             char(32)     not null,
    user_id             integer      not null,
    metric              varchar(256) not null,
    splitter            json         not null,
    ml_task             json         not null,
    specific_task_token varchar(256) not null,
    train_set_id        char(32)     not null,
    test_set_id         char(32)     not null,
    train_label_id      char(32)     not null,
    test_label_id       char(32)     not null,
    sub_sample_indexes  json         not null,
    sub_feature_indexes json         not null,
    task_metadata       json         not null,
    create_time         timestamp    not null,
    modify_time         timestamp    not null,
    constraint task_pkey
        primary key (task_id, user_id)
);
create table trial
(
    trial_id             serial       not null
        constraint trial_pkey
            primary key,
    user_id              integer      not null,
    config_id            char(32)     not null,
    run_id               char(256)    not null,
    instance_id          char(128)    not null,
    experiment_id        integer      not null,
    task_id              char(32)     not null,
    hdl_id               char(32)     not null,
    estimator            varchar(256) not null,
    loss                 real         not null,
    losses               json         not null,
    test_loss            json         not null,
    all_score            json         not null,
    all_scores           json         not null,
    test_all_score       json         not null,
    models_path          text         not null,
    final_model_path     text         not null,
    y_info_path          text         not null,
    additional_info      json         not null,
    dict_hyper_param     json         not null,
    cost_time            real         not null,
    status               varchar(32)  not null,
    failed_info          text         not null,
    warning_info         text         not null,
    intermediate_results json         not null,
    start_time           timestamp    not null,
    end_time             timestamp    not null
);
create index trial_task_id
    on trial (task_id);