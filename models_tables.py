from sqlalchemy import Column, Integer, String, Date, SmallInteger, PrimaryKeyConstraint
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class TLS201_APPLN(Base):
    __tablename__ = 'TLS201_APPLN'

    appln_id = Column(Integer, primary_key=True)
    appln_auth = Column(String(2), nullable=False)
    appln_nr = Column(String(15), nullable=False)
    appln_kind = Column(String(2), nullable=False)
    appln_filing_date = Column(Date, nullable=False)
    appln_filing_year = Column(SmallInteger, nullable=False)
    appln_nr_epodoc = Column(String(20), nullable=False)
    appln_nr_original = Column(String(100), nullable=False)
    ipr_type = Column(String(2), nullable=False)
    receiving_office = Column(String(2), nullable=False)
    internat_appln_id = Column(Integer, nullable=False)
    int_phase = Column(String(1), nullable=False)
    reg_phase = Column(String(1), nullable=False)
    nat_phase = Column(String(1), nullable=False)
    earliest_filing_date = Column(Date, nullable=False)
    earliest_filing_year = Column(SmallInteger, nullable=False)
    earliest_filing_id = Column(Integer, nullable=False)
    earliest_publn_date = Column(Date, nullable=False)
    earliest_publn_year = Column(SmallInteger, nullable=False)
    earliest_pat_publn_id = Column(Integer, nullable=False)
    granted = Column(String(1), nullable=False)
    docdb_family_id = Column(Integer, nullable=False)
    inpadoc_family_id = Column(Integer, nullable=False)
    docdb_family_size = Column(SmallInteger, nullable=False)
    nb_citing_docdb_fam = Column(SmallInteger, nullable=False)
    nb_applicants = Column(SmallInteger, nullable=False)
    nb_inventors = Column(SmallInteger, nullable=False)

class TLS202_APPLN_TITLE(Base):
    __tablename__ = 'TLS202_APPLN_TITLE'

    appln_id = Column(Integer, primary_key=True)
    appln_title_lg = Column(String(2), nullable=False)
    appln_title = Column(String, nullable=False)

class TLS204_APPLN_PRIOR(Base):
    __tablename__ = 'TLS204_APPLN_PRIOR'

    appln_id = Column(Integer, primary_key=True)
    prior_appln_id = Column(Integer, nullable=False)
    prior_appln_seq_nr = Column(SmallInteger, nullable=False)

class TLS206_PERSON(Base):
    __tablename__ = 'TLS206_PERSON'

    person_id = Column(Integer, primary_key=True)
    person_name = Column(String, nullable=False)
    person_name_orig_lg = Column(String, nullable=False)
    person_address = Column(String, nullable=False)
    person_ctry_code = Column(String(2), nullable=False)
    nuts = Column(String(5), nullable=False)
    nuts_level = Column(SmallInteger, nullable=False)
    doc_std_name_id = Column(Integer, nullable=False)
    doc_std_name = Column(String, nullable=False)
    psn_id = Column(Integer, nullable=False)
    psn_name = Column(String, nullable=False)
    psn_level = Column(SmallInteger, nullable=False)
    psn_sector = Column(String(50), nullable=False)
    han_id = Column(Integer, nullable=False)
    han_name = Column(String, nullable=False)
    han_harmonized = Column(Integer, nullable=False)

class TLS207_PERS_APPLN(Base):
    __tablename__ = 'TLS207_PERS_APPLN'

    person_id = Column(Integer, nullable=False)
    appln_id = Column(Integer, nullable=False)
    applt_seq_nr = Column(SmallInteger, nullable=False)
    invt_seq_nr = Column(SmallInteger, nullable=False)

    # Define a composite primary key using both person_id and appln_id
    __table_args__ = (
        PrimaryKeyConstraint('person_id', 'appln_id'),
    )

class TLS226_PERSON_ORIG(Base):
    __tablename__ = 'TLS226_PERSON_ORIG'

    person_orig_id = Column(Integer, primary_key=True)
    person_id = Column(Integer, nullable=False)
    source = Column(String(5), nullable=False)
    source_version = Column(String(10), nullable=False)
    name_freeform = Column(String(1000), nullable=False)
    person_name_orig_lg = Column(String(1000), nullable=False)
    last_name = Column(String(1000), nullable=False)
    first_name = Column(String(1000), nullable=False)
    middle_name = Column(String(1000), nullable=False)
    address_freeform = Column(String, nullable=False)
    address_1 = Column(String(1000), nullable=False)
    address_2 = Column(String(1000), nullable=False)
    address_3 = Column(String(1000), nullable=False)
    address_4 = Column(String(1000), nullable=False)
    address_5 = Column(String(1000), nullable=False)
    street = Column(String(1000), nullable=False)
    city = Column(String(500), nullable=False)
    zip_code = Column(String(30), nullable=False)
    state = Column(String(2), nullable=False)
    person_ctry_code = Column(String(2), nullable=False)
    residence_ctry_code = Column(String(2), nullable=False)
    role = Column(String(2), nullable=False)

 
class TLS211_PAT_PUBLN(Base):
    __tablename__ = 'TLS211_PAT_PUBLN'
   
    pat_publn_id = Column(Integer, nullable=False)           
    publn_auth = Column(String(2), nullable=False)           
    publn_nr = Column(String(15), nullable=False)           
    publn_nr_original = Column(String(100), nullable=False) 
    publn_kind = Column(String(2), nullable=False)          
    appln_id = Column(Integer, nullable=False)              
    publn_date = Column(Date, nullable=False)               
    publn_lg = Column(String(2), nullable=False)            
    publn_first_grant = Column(String(1), nullable=False)   
    publn_claims = Column(SmallInteger, nullable=False)     

    # Define the primary key as pat_publn_id
    __table_args__ = (
        PrimaryKeyConstraint('pat_publn_id'),
    )

class TLS224_APPLN_CPC(Base):
    __tablename__ = 'TLS224_APPLN_CPC'
    
    appln_id = Column(Integer, nullable=False)
    cpc_class_symbol = Column(String(19), nullable=False)

    __table_args__ = (
        PrimaryKeyConstraint('appln_id', 'cpc_class_symbol', name='pk_tls224_appln_cpc'),
    )
	 
class TLS209_APPLN_IPC(Base):
    __tablename__ = 'TLS209_appln_ipc'

    appln_id = Column(Integer, nullable=False)
    ipc_class_symbol = Column(String(15), nullable=False)
    ipc_class_level = Column(String(1), nullable=False)
    ipc_version = Column(Date, nullable=False)
    ipc_value = Column(String(1), nullable=False)
    ipc_position = Column(String(1), nullable=False)
    ipc_gener_auth = Column(String(2), nullable=False)
    
    __table_args__ = (
            PrimaryKeyConstraint('appln_id', 'ipc_class_symbol', name='pk_tls209_appln_ipc'),
        )
	 
