
'''
create account, login: APP_USER, APP_USER_LOGIN, APP_USER_LOGIN_DETAIL, MOBILE_DEVICE

booking: USER_BOOKING (HOURLY, DAILY, OVERNIGHT)
instant booking: BOOKING_INSTANT
ROOM_TYPE, FLASH_SALE_HISTORY, FLASH_SALE_BOOKING
notice: 
history: USER_ACCESS
'''

get_booking = '''
select booking.*,
hotel.TYPE AS HOTEL_TYPE,
hotel.COMPANY_NAME,
hotel.NAME AS HOTEL_NAME,
hotel.ADDRESS,
hotel.HOTEL_STATUS,
hotel.CODE,
hotel.ORIGIN,
history.PRICE_ONE_DAY,
history.PRICE_OVERNIGHT,
history.PRICE_FIRST_HOURS,
history.FIRST_HOURS,
history.PRICE_ADDITIONAL_HOURS,
history.ADDITIONAL_HOURS,
IF(booking.FLASH_SALE_HISTORY_SN IS NOT NULL, IF(flashSaleHistory.NAME_SHORT_NAME IS NOT NULL,flashSaleHistory.NAME_SHORT_NAME, roomType.NAME), roomType.NAME) AS ROOM_TYPE_NAME,
roomType.SHORT_NAME,
coupon.DISCOUNT_TYPE,coupon.DISCOUNT,
coupon.PROMOTION_SN,
coupon.TITLE AS COUPON_NAME,
coupon.COUPON_SN,
coupon.CODE AS COUPON_CODE,
appUser.NICK_NAME,
appUser.MEMBER_ID,
appUser.EMAIL,
appUser.REGISTER_TIME,
appUser.GENDER,
appUser.BIRTHDAY,
province.NAME AS PROVINCE_NAME,
district.NAME AS DISTRICT_NAME,
IF(booking.FLASH_SALE_HISTORY_SN IS NOT NULL, 2, 1) AS MODE,
cancellationPolicy.REFUNDING_AMOUNT AS REFUNDING_AMOUNT,
guestBooking.FIRST_NAME AS FIRST_NAME_GUEST,
guestBooking.LAST_NAME AS LAST_NAME_GUEST,
guestBooking.COUNTRY_CODE AS COUNTRY_CODE_GUEST,
guestBooking.MOBILE AS MOBILE_GUEST from  USER_BOOKING booking
LEFT JOIN HOTEL hotel ON booking.HOTEL_SN = hotel.SN
LEFT JOIN ROOM_TYPE roomType ON booking.ROOM_TYPE_SN = roomType.SN
LEFT JOIN APP_USER appUser ON booking.APP_USER_SN = appUser.SN
LEFT JOIN ROOM_TYPE_HISTORY history ON booking.ROOM_TYPE_HISTORY_SN = history.SN
LEFT JOIN CANCELLATION_POLICY cancellationPolicy ON cancellationPolicy.USER_BOOKING_SN = booking.SN
LEFT JOIN GUEST_BOOKING guestBooking ON guestBooking.USER_BOOKING_SN = booking.SN
LEFT JOIN USER_BOOKING_REASON_CANCEL userBookingReasonCancel ON userBookingReasonCancel.USER_BOOKING_SN = booking.SN
LEFT JOIN (SELECT issued.*,coupon.DISCOUNT_TYPE,coupon.DISCOUNT,coupon.PROMOTION_SN AS
    PROMOTION_SN,coupon.TITLE,coupon.CODE
    FROM COUPON_ISSUED issued,COUPON coupon
    WHERE coupon.SN = issued.COUPON_SN  )coupon ON booking.COUPON_ISSUED_SN = coupon.SN
LEFT JOIN PROVINCE province ON province.SN = hotel.PROVINCE_SN  LEFT JOIN DISTRICT district ON district.SN = hotel.DISTRICT_SN
LEFT JOIN FLASH_SALE_HISTORY flashSaleHistory ON flashSaleHistory.BOOKING_SN = booking.SN  where `booking`.`HOTEL_SN` != 467 and  (booking.BOOKING_STATUS = 1 OR (booking.BOOKING_STATUS = 0 AND booking.PAYMENT_PROVIDER = 20))  and `booking`.`CHECK_IN_DATE_PLAN` >=  "2022-01-01 00:00:00"  order by IF((booking.BOOKING_STATUS=1 || booking.BOOKING_STATUS=2) && userBookingReasonCancel.PORTAL_REASON_CANCELLATION_SN IS NOT NULL, 1, 0) desc, booking.HOTEL_SN ASC,booking.CREATE_TIME DESC;

'''

get_lst_hotel = """
select `H`.`ALLOW_EXTRA_FEE` as `allowExtraFee`, `H`.`SN` as `sn`, `H`.`NAME` as `name`, `H`.`ADDRESS` as `address`, `H`.`LONGITUDE` as `longitude`, `H`.`LATITUDE` as `latitude`, `H`.`COUNTRY_SN` as `countrySn`, `H`.`PROVINCE_SN` as `provinceSn`, `H`.`DISTRICT_SN` as `districtSn`, `H`.`AREA_CODE` as `areaCode`, `H`.`PHONE` as `phone`, `H`.`NEW_HOTEL` as `newHotel`, `H`.`HOT_HOTEL` as `hotHotel`, `H`.`HAS_PROMOTION` as `hasPromotion`, `H`.`HAS_GIFT` as `hasGift`, `H`.`HAS_BONUS_HOUR` as `hasBonusHour`, `H`.`DESCRIPTION` as `description`, `H`.`FOLDER_IMAGE` as `folderImage`, `H`.`TOTAL_REVIEW` as `totalReview`, `H`.`AVERAGE_MARK` as `averageMark`, `H`.`TOTAL_FAVORITE` as `totalFavorite`, `H`.`CREATE_TIME` as `createTime`, `H`.`CONTRACT_DATE` as `contractDate`, `H`.`END_CONTRACT_DATE` as `endContractDate`, `H`.`NEW_DATE` as `newDate`, `H`.`HOTEL_STATUS` as `hotelStatus`, `H`.`ROOM_AVAILABLE` as `roomAvailable`, `H`.`COMMISSION` as `commission`, `H`.`LOWEST_PRICE` as `lowestPrice`, `H`.`FIRST_HOURS` as `firstHours`, `H`.`LOWEST_PRICE_OVERNIGHT` as `lowestPriceOvernight`, `H`.`LOWEST_ONE_DAY` as `lowestOneDay`, `H`.`BANK_NAME` as `bankName`, `H`.`BANK_ACCOUNT` as `bankAccount`, `H`.`BENEFICIARY` as `beneficiary`, `H`.`BANK_BRANCH` as `bankBranch`, `H`.`COUNT_EXIF_IMAGE` as `countExifImage`, `H`.`LAST_COMMENT` as `lastComment`, `H`.`MAX_BOOKING` as `maxBooking`, `H`.`CODE` as `code`, `H`.`HOTEL_GROUP_SN` as `hotelGroupSn`, `H`.`DISCOUNT` as `discount`, `H`.`OVERNIGHT_ORIGIN` as `overnightOrigin`, `H`.`FIRST_HOURS_ORIGIN` as `firstHoursOrigin`, `H`.`ONE_DAY_ORIGIN` as `oneDayOrigin`, `H`.`STYLE` as `style`, `H`.`TYPE` as `type`, `H`.`NUM_NEW_REVIEW` as `numNewReview`, `H`.`HASH_TAG` as `hashtag`, `H`.`MIN_DISCOUNT_DATE` as `minDiscountDate`, `H`.`TOTAL_VIEW` as `totalView`, `H`.`NUM_TO_REDEEM` as `numToRedeem`, `H`.`NUM_NOT_CONFIRMED` as `numNotConfirmed`, `H`.`SALE_IN_CHARGE_SN` as `saleInChargeSn`, `H`.`BIZ_IN_CHARGE_SN` as `bizInChargeSn`, `H`.`CREATE_STAFF_SN` as `createStaffSn`, `H`.`TOP` as `top`, `H`.`TOP_END_DATE` as `topEndDate`, `H`.`EN_DESCRIPTION` as `enDescription`, `H`.`DISPLAY` as `display`, `H`.`AVERAGE_MARK_FACILITY` as `averageMarkFacility`, `H`.`AVERAGE_MARK_CLEAN` as `averageMarkClean`, `H`.`AVERAGE_MARK_SERVICE` as `averageMarkService`, `H`.`NUM_OF_STAR_REVIEW` as `numOfStarReview`, `H`.`IMAGE_PATH` as `imagePath`, `H`.`EXTRA_FEE` as `extraFee`, `H`.`TAX_ID` as `taxId`, `H`.`BUSINESS_LICENSE` as `businessLicense`, `H`.`COMPANY_NAME` as `companyName`, `H`.`COMPANY_ADDRESS` as `companyAddress`, `H`.`REP_NAME` as `repName`, `H`.`REP_POSITION` as `repPosition`, `H`.`REP_TEL` as `repTel`, `H`.`REP_EMAIL` as `repEmail`, `H`.`NEW_CONTRACTED_TYPE` as `newContractedType`, `H`.`G2J_CERTIFIED` as `g2jCertified`, `H`.`ORIGIN` as `origin`, `H`.`POLICY` as `policy`, `H`.`POLICY_EN` as `policyEn`,
                JSON_OBJECT('sn', SIC.SN, 'fullName', SIC.FULL_NAME) as saleInCharge,
                JSON_OBJECT('sn', BIC.SN, 'fullName', BIC.FULL_NAME) as bizInCharge,
                (
                    SELECT (JSON_OBJECT(
                        'sn', HG.SN,
                        'name', HG.NAME
                    ))
                    from HOTEL_GROUP as HG
                    where HG.SN = H.HOTEL_GROUP_SN
                ) as hotelGroup,
                CASE
                    WHEN H.ORIGIN = 1 THEN
                        JSON_OBJECT(
                            'checkInTime', (SELECT IFNULL(CAST(S.NUMDATA4 AS UNSIGNED INTEGER), NULL) FROM SETTING AS S WHERE S.HOTEL_SN = H.SN AND COMMON_NO = 0007 AND CLASS_NO = 03),
                            'checkoutTime', (SELECT IFNULL(CAST(S.NUMDATA1 AS UNSIGNED INTEGER), NULL) FROM SETTING AS S WHERE S.HOTEL_SN = H.SN AND COMMON_NO = 0007 AND CLASS_NO = 03),
                            'overStartTime', (SELECT IFNULL(CAST(S.NUMDATA2 AS UNSIGNED INTEGER), NULL) FROM SETTING AS S WHERE S.HOTEL_SN = H.SN AND COMMON_NO = 0007 AND CLASS_NO = 03),
                            'overEndTime', (SELECT IFNULL(CAST(S.NUMDATA3 AS UNSIGNED INTEGER), NULL) FROM SETTING AS S WHERE S.HOTEL_SN = H.SN AND COMMON_NO = 0007 AND CLASS_NO = 03),
                            'payment', (SELECT IFNULL(CAST(S.CHARDATA3 AS UNSIGNED INTEGER), NULL) FROM SETTING AS S WHERE S.HOTEL_SN = H.SN AND COMMON_NO = 0007 AND CLASS_NO = 03)
                        )
                    WHEN H.ORIGIN = 2 THEN
                        JSON_OBJECT(
                            'checkInTime', (SELECT IFNULL(CAST(S.CHARDATA1 AS UNSIGNED INTEGER), '14') FROM SETTING AS S WHERE S.HOTEL_SN = H.SN AND COMMON_NO = 0017 AND CLASS_NO = 01),
                            'checkoutTime', (SELECT IFNULL(CAST(S.CHARDATA4 AS UNSIGNED INTEGER), '12') FROM SETTING AS S WHERE S.HOTEL_SN = H.SN AND COMMON_NO = 0017 AND CLASS_NO = 01),
                            'overStartTime', NULL,
                            'overEndTime', NULL,
                            'payment', (SELECT IFNULL(CAST(S.CHARDATA2 AS UNSIGNED INTEGER), '3') FROM SETTING AS S WHERE S.HOTEL_SN = H.SN and COMMON_NO = 0017 AND CLASS_NO = 03)
                        )
                END as setting,
                (
                    select JSON_ARRAYAGG(JSON_OBJECT(
                        'sn', S.SN,
                        'fullName', S.FULL_NAME,
                        'userId', S.USER_ID,
                        'mobile', S.MOBILE,
                        'email', S.EMAIL,
                        'status', S.STATUS,
                        'receiveSms', S.RECEIVE_SMS,
                        'regionsMgt', S.REGIONS_MGT,
                        'role', JSON_OBJECT('sn', R.SN, 'name', R.NAME)
                    ))
                    from STAFF as S
                    inner join ROLE as R on S.ROLE_SN = R.SN
                    where S.HOTEL_SN = H.SN
                ) as staffs
             from `HOTEL` as `H` left join `STAFF` as `BIC` on `BIC`.`SN` = `H`.`BIZ_IN_CHARGE_SN` left join `STAFF` as `SIC` on `SIC`.`SN` = `H`.`SALE_IN_CHARGE_SN` where `H`.`SN` != 1 and H.PROVINCE_SN IN (SELECT p.SN
                                    FROM (SELECT SN, REGIONS FROM PROVINCE) AS p
                                    JOIN (SELECT REGIONS_MGT FROM STAFF WHERE SN = 3662) AS s ON s.REGIONS_MGT LIKE concat('%%',p.REGIONS,'%%')) order by createTime desc;
"""

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
where YEAR(CREATE_TIME) >= 2019 and HOTEL_SN != 467
"""

get_hotel ="""
select SN, NAME, ADDRESS, LONGITUDE, LATITUDE, COUNTRY_SN, PROVINCE_SN, DISTRICT_SN, AVERAGE_MARK_FACILITY, \
            AVERAGE_MARK_CLEAN, AVERAGE_MARK_SERVICE, NUM_OF_STAR_REVIEW, STAR_RATING
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