import psycopg2
import sys
import numpy as np
import math
import dateutil.parser
import datetime,time

from grand_spectrum_core.admx_db_datatypes import PowerSpectrum

class ADMXDB:
    """A connection to the admx database"""

    def __init__(self):
        self.hostname="localhost"
        self.dbname="admx"
        self.port="5432"

    def get_na_scan(self,**kwargs):
        """Retrieves network analyzer scans, returns array of spectra.  accepted keywords are
           table (if you want the sidecar table instead set equal to sidecar_na_log)
           notes (select by notes)
           start_time
           stop_time
           limit (max number of results)"""
        to_select=["timestamp","start_frequency","stop_frequency","frequency_resolution","fft_data","notes"]
        conditions=[]
        limit=10
        thetable="na_log"
        if 'table' in kwargs:
            thetable=kwargs['table']
        if 'notes' in kwargs:
            conditions.append("notes='"+kwargs['notes']+"'")
        if 'naid' in kwargs:
            if type(kwargs['naid']) in (tuple,list):
                fullcond=""
                for x in kwargs['naid']:
                    if fullcond!="":
                        fullcond=fullcond+" OR "
                    fullcond=fullcond+"na_log_id='"+str(x)+"'"
                conditions.append(fullcond)
            else:
                conditions.append("na_log_id='"+str(kwargs["naid"])+"'")
     
        if 'start_time' in kwargs:
            if "'" in kwargs['start_time']:
                conditions.append("timestamp>"+kwargs['start_time'])
            else:
                conditions.append("timestamp>'"+kwargs['start_time']+"'")
        if 'stop_time' in kwargs:
            if "'" in kwargs['stop_time']:
                conditions.append("timestamp<"+kwargs['stop_time'])
            else:
                conditions.append("timestamp<'"+kwargs['stop_time']+"'")
        if 'limit' in kwargs:
            limit=kwargs['limit']
        if 'additional_condition' in kwargs:
            conditions.append(kwargs['additional_condition'])
        complex_result='complex_data' in kwargs
    #   special metadata for sidecar
        if thetable=="sidecar_na_log":
            to_select.append("rod_pos_calibrated")

        query_string=self.build_simple_select_query(thetable,to_select,conditions,limit)
        records=self.send_admxdb_query(query_string)
        ret=[]
        for i in range(len(records)):
            my_scan_fft=records[i][4]
            my_scan_start_freq=records[i][1]
            my_scan_stop_freq=records[i][2]
            my_scan_freq_resolution=records[i][3]
            powers=[]
            if complex_result:
                for j in range(int(len(my_scan_fft)/2)):
                    powers.append( complex( my_scan_fft[2*j],my_scan_fft[2*j+1] ) )
            else:
                for j in range(int(len(my_scan_fft)/2)):
                    powers.append( (my_scan_fft[2*j]*my_scan_fft[2*j] + my_scan_fft[2*j+1]*my_scan_fft[2*j+1] ) )
            my_metadata={}
            for j in range(len(to_select)):
                if to_select[j] != "fft_data": #don't want that giant array there
                    my_metadata[to_select[j]]=records[i][j]
#ret.append(Spectrum(powers,my_scan_start_freq,my_scan_stop_freq,metadata=my_metadata))
            ret.append(PowerSpectrum(powers,my_scan_start_freq,my_scan_stop_freq,metadata=my_metadata))
        return ret

    def get_dig_scan(self,**kwargs):
        """Retrieves digitizer scans, returns as array of Spectra.  accepted kewords are
           notes
           start_time
           stop_time
           limit
           channel (if you want sidecar instead)"""
        to_select=["timestamp","start_frequency_channel_one","stop_frequency_channel_one","power_spectrum_channel_one","notes","switchbox_attenuation_channel_1","digitizer_log_id","lo_frequency_channel_one"]
        conditions=[]
        limit=10
        thetable="digitizer_log"
        if 'channel' in kwargs:
           if kwargs["channel"]=="sidecar":
               thetable="sidecar_digitizer_log"
               to_select=["timestamp","start_frequency_sidecar","stop_frequency_sidecar","power_spectrum_sidecar","notes","digitizer_log_id","lo_frequency_sidecar"]

        if 'notes' in kwargs:
            conditions.append("notes='"+kwargs['notes']+"'")
        if 'digid' in kwargs:
            if type(kwargs['digid']) in (tuple,list):
                fullcond=""
                for x in kwargs['digid']:
                    if fullcond!="":
                        fullcond=fullcond+" OR "
                    fullcond=fullcond+"digitizer_log_id='"+str(x)+"'"
                conditions.append(fullcond)
            else:
                conditions.append("digitizer_log_id='"+kwargs["digid"]+"'")
            
        if 'start_time' in kwargs:
            if "'" in kwargs['start_time']:
                conditions.append("timestamp>"+kwargs['start_time'])
            else:
                conditions.append("timestamp>'"+kwargs['start_time']+"'")
        if 'stop_time' in kwargs:
            if "'" in kwargs['stop_time']:
                conditions.append("timestamp<"+kwargs['stop_time'])
            else:
                conditions.append("timestamp<'"+kwargs['stop_time']+"'")
        if 'limit' in kwargs:
            limit=str(kwargs['limit'])
        query_string=self.build_simple_select_query(thetable,to_select,conditions,limit)
        print(query_string)
        records=self.send_admxdb_query(query_string)
        ret=[]
        for i in range(len(records)):
            if thetable!="digitizer_log": #sidecar mode
                lo_freq=records[i][6]
                #print(lo_freq)
                ret.append(PowerSpectrum(records[i][3],records[i][1]+lo_freq,records[i][2]+lo_freq,metadata={"timestamp": records[i][0],"notes": records[i][4],"digitizer_log_id":records[i][5]}))
            else: #main channel mode              
                lo_freq=records[i][7]
                #print(lo_freq)
                if lo_freq is None:
                    print("Warning, no lo for {}, skipping".format(records[i][0]))
                    continue
                if records[i][1] is None:
                    print("Warning, bad scan for {}, skipping".format(records[i][0]))
                    continue
                ret.append(PowerSpectrum(records[i][3],records[i][1]+lo_freq,records[i][2]+lo_freq,metadata={"timestamp": records[i][0],"notes": records[i][4],"attenuation": records[i][5],"digitizer_log_id":records[i][6]}))
        return ret

    def send_admxdb_query(self,query_string):
        """This sends a query to the ADMX database and returns all matching results"""
        cursor=self.send_admxdb_query_but_dont_fetch(query_string)
        records=cursor.fetchall()
        return records

    def send_admxdb_query_but_dont_fetch(self,query_string):
        """This sends a query to the ADMX database and returns the cursor from which results can be fetched
            TODO remove username and password from hardcoded here"""
#conn_string="host='admxdatastore.npl.washington.edu' dbname='admx' user='admx_reader' password='iheartdm'"
        conn_string="host='"+self.hostname+"' dbname='"+self.dbname+"' user='admx_reader' password='iheartdm'"
        if self.port:
            conn_string=conn_string+" port='"+str(self.port)+"'"
        conn=psycopg2.connect(conn_string)
        cursor=conn.cursor()
        cursor.execute(query_string)
        return cursor


    def build_simple_select_query(self,from_table,what_to_select,conditions,limit):
        """Given a table name, colums to select, conditions as strings, 
            and the limit on number of entries to return, 
            returns an array of ( arrays of the column values )"""
        query_string="SELECT "
        first=True
        for x in what_to_select:
            if first==False:
                query_string=query_string+","
            query_string=query_string+x
            first=False
        first=True
        query_string=query_string+" FROM "+from_table
        for x in conditions:
            if first==True:
                query_string=query_string+" WHERE "
            else:
                query_string=query_string+" AND "
            query_string=query_string+x
            first=False
        query_string=query_string+" ORDER BY timestamp DESC"
        if limit:
            query_string=query_string+" LIMIT "+str(limit)
        return query_string

    def get_sensor_values(self,sensor_name,start_time,stop_time,**kwargs):
        """get an array of sensor values with a certain name between start and stop times.  Returns two arrays, one of times, one of calibrated values"""
        query_string="select sensor_name,timestamp,raw_value,calibrated_value from sensors_log_double where timestamp>'"+start_time+"' AND timestamp<'"+stop_time+"' AND sensor_name='"+sensor_name+"'"
        query_string=query_string+" order by timestamp asc"
        records=self.send_admxdb_query(query_string)
        thetimes=[]
        thevals=[]
        for x in records:
            thetimes.append((x[1]))
            thevals.append(x[3])
        if 'start_time_2' in kwargs:
            start_time_2 = kwargs['start_time_2']
            stop_time_2 = kwargs['stop_time_2']
            query_string="select sensor_name,timestamp,raw_value,calibrated_value from sensors_log_double where timestamp>'"+start_time_2+"' AND timestamp<'"+stop_time_2+"' AND sensor_name='"+sensor_name+"'"
            query_string=query_string+" order by timestamp asc"
            records_2=self.send_admxdb_query(query_string)
            for x in records_2:
                thetimes.append((x[1]))
                thevals.append(x[3])
        return thetimes,thevals

    #Added by Jimmy to also retrieve the error in Q which Dan calculated from the fitting of transmission scans
    def get_sensor_values_and_errors(self,sensor_name,start_time,stop_time):
        """get an array of sensor values with a certain name between start and stop times.  Returns two arrays, one of times, one of calibrated values"""
        query_string="select sensor_name,timestamp,raw_value,calibrated_value,calibrated_err from sensors_log_double where timestamp>'"+start_time+"' AND timestamp<'"+stop_time+"' AND sensor_name='"+sensor_name+"'"
        query_string=query_string+" order by timestamp asc"
        records=self.send_admxdb_query(query_string)
        thetimes=[]
        thevals=[]
        theerrs=[] #pronounced "the errs" -JS
        for x in records:
#thetimes.append(dateutil.parser.parse(x[1]))
            thetimes.append((x[1]))
            thevals.append(x[3])
            theerrs.append(x[4])
        return thetimes,thevals,theerrs
    

    def get_plot_by_name(self,name):
        """Get a plot off the database with as specific name"""
        query_string="select plot_name,timestamp,x_points,y_points,z_points,x_units,y_units,z_units,notes from plots where plot_name='"+name+"' order by timestamp desc limit 1"
        print(query_string)
        records=self.send_admxdb_query(query_string)
        #print records
        return records[0][2],records[0][3],records[0][4],{"name": records[0][0],"timestamp":records[0][1],"x_units":records[0][5],"y_units":records[0][6],"z_units":records[0][7]}

    def get_sag_table(self,frequency_range,time_range):
        '''download data from the synthetic injection record, possibly including a frequency range or time range []'''
        query_string="select timestamp,arb_power,output_freq,signal_type,scan_number FROM saglog"
        wh=False
        if frequency_range:
            if wh==False:
                query_string+=" WHERE"
                wh=True
            query_string+=" output_freq > '"+str(frequency_range[0])+"' AND output_freq < '"+str(frequency_range[1])+"'"
        if time_range:
            if wh==False:
                query_string+=" WHERE"
                wh=True
            else:
                query_string+=" AND"
            query_string+=" timestamp > '"+time_range[0].isoformat()+"' AND timestamp < '"+time_range[1].isoformat()+"'"
        records=self.send_admxdb_query(query_string)
        response={}
        #timestamps=[]
        #freqs=[]
        #pows=[]
        #types=[]
        #scan_nos=[]
        for i in range(len(records)):
            response[records[i][4]]={"timestamp": records[i][0],"freq": records[i][2],"power": math.pow(10.0,records[i][1]/10.0),"signal_type": records[i][3]}
            #timestamps.append(records[i][0])
            #freqs.append(records[i][2])
            #pows.append(records[i][1])
            #types.append(records[i][3])
            #scan_nos.append(records[i][4])
        #return timestamps,freqs,pows,types,scan_nos
        return response

            
