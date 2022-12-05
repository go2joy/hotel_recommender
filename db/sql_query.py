
# select count(COMMISSION_AMOUNT) 
# from USER_BOOKING
# WHERE YEAR(CREATE_TIME)= 2022

get_app_user = """
select SN, ADDRESS, GENDER, BIRTHDAY, STATUS, LAST_OPEN_APP, NUM_BOOKING, NUM_CHECKIN, REGISTER_BY, \
            REGISTER_TIME, NUM_OPEN_APP
from APP_USER          
"""

get_user_booking = """
select SN, HOTEL_SN,TOTAL_AMOUNT, ROOM_TYPE_SN,CHECK_IN_DATE_PLAN, BOOKING_STATUS, TYPE, \
            APP_USER_SN, END_DATE, START_TIME, END_TIME, CREATE_TIME, COMMISSION_AMOUNT
from USER_BOOKING
where HOTEL_SN != 467
"""

get_hotel ="""
select SN, NAME, LONGITUDE, LATITUDE, COUNTRY_SN, PROVINCE_SN, DISTRICT_SN, ADDRESS
from HOTEL            
"""

get_hotel_setting ="""
select * from V_HOTEL_SETTING

"""

get_room_type = """
select SN, HOTEL_SN,FIRST_HOURS, PRICE_FIRST_HOURS, PRICE_ADDITIONAL_HOURS, PRICE_OVERNIGHT, PRICE_ONE_DAY, \
            SQUARE, BED_TYPE, NUM_OF_ROOM, MAX_BOOKING, STATUS, ONE_DAY_ORIGIN, FIRST_HOURS_ORIGIN, ADDITIONAL_ORIGIN, ADDITIONAL_HOURS
from ROOM_TYPE
"""

get_province = """
select SN, NAME
from PROVINCE
"""

get_district = """
select SN, NAME
from DISTRICT
"""

lst_query = [get_user_booking, get_hotel]
# lst_query = [get_user_booking, get_hotel, get_hotel_setting, get_province, get_district, get_app_user, get_room_type]