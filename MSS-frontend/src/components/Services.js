import React from 'react';
import Daterange from './Daterange';

const Services = () => {

    return (
        <div id="services" className="bg-gray-100 py-12" >
            <section data-aos="zoom-in-down">
                <div className="my-4 py-4">
                    <h2 className="my-2 text-center text-3xl text-blue-900 uppercase font-bold">Fraction Dashboard</h2>
                    <br />
                    <h6 className="my-2 text-center text-xl text-blue-900 uppercase font-bold">HUMAN DETECTION DETAILS</h6>
                    {/* <h6 align='center'>Dashboard view of Human detected data and time  </h6> */}
                </div>
            </section>
            < Daterange />
        </div>
    )
}

export default Services;