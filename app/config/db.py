from orator import DatabaseManager

# Local Connection
config = {
    'mysql': {
        'driver'    : 'mysql',
        'host'      : 'localhost',
        'database'  : 'aplikasi_lutfiana_svm',
        'user'      : 'root',
        'password'  : '',
        'prefix'    : '',
        'charset'   : 'utf8mb4'
    }
}


db = DatabaseManager(config)
