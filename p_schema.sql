-- Switch to the closetx database
\c closetx;

-- Drop the tables if they exist
DROP TABLE IF EXISTS apparel CASCADE;
DROP TABLE IF EXISTS "user" CASCADE;

-- Create the "user" table
CREATE TABLE "user" (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) NULL,
    email VARCHAR(100) NULL UNIQUE,
    password VARCHAR(255) NULL
);

-- Create the "apparel" table
CREATE TABLE apparel (
    id SERIAL PRIMARY KEY,
    "user" INT,
    uri VARCHAR(255),
    FOREIGN KEY ("user") REFERENCES "user"(id),
    type INT
);
