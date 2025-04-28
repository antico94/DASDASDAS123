-- Create the logs schema if it does not exist
IF NOT EXISTS (SELECT 1 FROM sys.schemas WHERE name = 'logs')
BEGIN
    EXEC('CREATE SCHEMA logs');
    PRINT 'Schema [logs] created.';
END
ELSE
BEGIN
    PRINT 'Schema [logs] already exists.';
END
GO -- Use GO to separate batches in SQL Server

-- Create the order_requests table in the logs schema
IF NOT EXISTS (SELECT 1 FROM sys.tables WHERE name = 'order_requests' AND schema_id = SCHEMA_ID('logs'))
BEGIN
    CREATE TABLE logs.order_requests (
        id INT IDENTITY(1,1) PRIMARY KEY,
        time DATETIME2 NOT NULL, -- DATETIME2 preferred over DATETIME
        type NVARCHAR(50) NOT NULL, -- NVARCHAR for Unicode support
        message NVARCHAR(MAX) NOT NULL, -- NVARCHAR(MAX) for TEXT equivalent

        symbol NVARCHAR(20),
        volume FLOAT, -- Use FLOAT or REAL for floating-point numbers
        price FLOAT,
        stop_loss FLOAT,
        take_profit FLOAT,
        ticket INT,
        strategy NVARCHAR(50)
    );
    PRINT 'Table [logs].[order_requests] created.';
END
ELSE
BEGIN
    PRINT 'Table [logs].[order_requests] already exists.';
END
GO

-- Create the order_executions table in the logs schema
IF NOT EXISTS (SELECT 1 FROM sys.tables WHERE name = 'order_executions' AND schema_id = SCHEMA_ID('logs'))
BEGIN
    CREATE TABLE logs.order_executions (
        id INT IDENTITY(1,1) PRIMARY KEY,
        time DATETIME2 NOT NULL,
        type NVARCHAR(50) NOT NULL,
        message NVARCHAR(MAX) NOT NULL,

        symbol NVARCHAR(20),
        volume FLOAT,
        price FLOAT,
        ticket INT,
        profit FLOAT,
        strategy NVARCHAR(50)
    );
    PRINT 'Table [logs].[order_executions] created.';
END
ELSE
BEGIN
    PRINT 'Table [logs].[order_executions] already exists.';
END
GO

-- Create the positions table in the logs schema
IF NOT EXISTS (SELECT 1 FROM sys.tables WHERE name = 'positions' AND schema_id = SCHEMA_ID('logs'))
BEGIN
    CREATE TABLE logs.positions (
        id INT IDENTITY(1,1) PRIMARY KEY,
        time DATETIME2 NOT NULL,
        type NVARCHAR(50) NOT NULL,
        message NVARCHAR(MAX) NOT NULL,

        symbol NVARCHAR(20),
        ticket INT,
        volume FLOAT,
        open_price FLOAT,
        current_price FLOAT,
        stop_loss FLOAT,
        take_profit FLOAT,
        profit FLOAT,
        strategy NVARCHAR(50)
    );
    PRINT 'Table [logs].[positions] created.';
END
ELSE
BEGIN
    PRINT 'Table [logs].[positions] already exists.';
END
GO

-- Create the account_snapshots table in the logs schema
IF NOT EXISTS (SELECT 1 FROM sys.tables WHERE name = 'account_snapshots' AND schema_id = SCHEMA_ID('logs'))
BEGIN
    CREATE TABLE logs.account_snapshots (
        id INT IDENTITY(1,1) PRIMARY KEY,
        time DATETIME2 NOT NULL,
        type NVARCHAR(50) NOT NULL, -- Although your model defaults to "SNAPSHOT", it's still nullable=False
        message NVARCHAR(MAX) NOT NULL,

        balance FLOAT,
        equity FLOAT,
        margin FLOAT,
        free_margin FLOAT,
        margin_level FLOAT,
        open_positions INT
    );
    PRINT 'Table [logs].[account_snapshots] created.';
END
ELSE
BEGIN
    PRINT 'Table [logs].[account_snapshots] already exists.';
END
GO

-- Create the errors table in the logs schema
IF NOT EXISTS (SELECT 1 FROM sys.tables WHERE name = 'errors' AND schema_id = SCHEMA_ID('logs'))
BEGIN
    CREATE TABLE logs.errors (
        id INT IDENTITY(1,1) PRIMARY KEY,
        time DATETIME2 NOT NULL,
        type NVARCHAR(50) NOT NULL, -- ERROR, WARNING
        message NVARCHAR(MAX) NOT NULL,

        component NVARCHAR(100),
        exception_type NVARCHAR(100),
        stacktrace NVARCHAR(MAX) -- NVARCHAR(MAX) for TEXT equivalent
    );
    PRINT 'Table [logs].[errors] created.';
END
ELSE
BEGIN
    PRINT 'Table [logs].[errors] already exists.';
END
GO

-- Create the events table in the logs schema
IF NOT EXISTS (SELECT 1 FROM sys.tables WHERE name = 'events' AND schema_id = SCHEMA_ID('logs'))
BEGIN
    CREATE TABLE logs.events (
        id INT IDENTITY(1,1) PRIMARY KEY,
        time DATETIME2 NOT NULL,
        type NVARCHAR(50) NOT NULL, -- INFO, DEBUG, SIGNAL, STRATEGY
        message NVARCHAR(MAX) NOT NULL,

        component NVARCHAR(100),
        details NVARCHAR(MAX) -- NVARCHAR(MAX) for TEXT equivalent
    );
    PRINT 'Table [logs].[events] created.';
END
ELSE
BEGIN
    PRINT 'Table [logs].[events] already exists.';
END
GO