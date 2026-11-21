import pytest

from attendance.db import init_db, make_engine, make_session_factory


@pytest.fixture
def db():
    engine = make_engine("sqlite:///:memory:")
    init_db(engine)
    session = make_session_factory(engine)()
    yield session
    session.close()
