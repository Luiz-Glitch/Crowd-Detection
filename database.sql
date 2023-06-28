CREATE DATABASE crowd_detection;

CREATE TABLE crowd_records (
    rec_id      INT                 NOT NULL    AUTO_INCREMENT   PRIMARY KEY,
    crowd_id    INT UNSIGNED        NOT NULL,
    size        INT UNSIGNED        NOT NULL,
    rec_time    DATETIME,
    image       VARCHAR(50));