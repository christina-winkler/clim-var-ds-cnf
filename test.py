import cdsapi

def retrieve_era5_data():
    c = cdsapi.Client()
    c.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': 'total_column_water',
            'year': '2023',
            'month': '01',
            'day': '01',
            'time': '00:00',
            'format': 'netcdf'
        },
        'total_water_content.nc'
    )

if __name__ == "__main__":
    retrieve_era5_data()
